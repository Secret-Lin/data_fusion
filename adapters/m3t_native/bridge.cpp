// Project-owned adapter around unmodified DLR M3T (MIT).
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/depth_modality.h>
#include <m3t/region_modality.h>
#include <m3t/tracker.h>
#include <omp.h>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using ColorArray = py::array_t<unsigned char, py::array::c_style>;
using DepthArray = py::array_t<unsigned short, py::array::c_style>;

static void require(bool ok, const char* stage) {
  if (!ok) throw std::runtime_error(std::string("M3T failed: ") + stage);
}

class ModelLog {
 public:
  explicit ModelLog(const std::filesystem::path& path) : file_(path, std::ios::app) {
    if (!file_) throw std::runtime_error("Cannot open M3T model setup log");
    out_ = std::cout.rdbuf(file_.rdbuf());
    err_ = std::cerr.rdbuf(file_.rdbuf());
  }
  ~ModelLog() { std::cout.rdbuf(out_); std::cerr.rdbuf(err_); }
 private:
  std::ofstream file_;
  std::streambuf *out_, *err_;
};

class MemoryColor : public m3t::ColorCamera {
 public:
  MemoryColor(const m3t::Intrinsics& k) : ColorCamera("rgb") { intrinsics_ = k; }
  bool SetUp() override { set_up_ = !image_.empty(); return set_up_; }
  bool UpdateImage(bool) override { return !image_.empty(); }
  void put(ColorArray a) {
    if (a.ndim() != 3 || a.shape(0) != intrinsics_.height ||
        a.shape(1) != intrinsics_.width || a.shape(2) != 3)
      throw std::invalid_argument("RGB must be BGR uint8 HxWx3 at native resolution");
    image_ = cv::Mat(intrinsics_.height, intrinsics_.width, CV_8UC3, a.mutable_data()).clone();
  }
};

class MemoryDepth : public m3t::DepthCamera {
 public:
  MemoryDepth(const m3t::Intrinsics& k) : DepthCamera("depth") {
    intrinsics_ = k; depth_scale_ = 0.001f;
  }
  bool SetUp() override { set_up_ = !image_.empty(); return set_up_; }
  bool UpdateImage(bool) override { return !image_.empty(); }
  void put(DepthArray a) {
    if (a.ndim() != 2 || a.shape(0) != intrinsics_.height || a.shape(1) != intrinsics_.width)
      throw std::invalid_argument("Depth must be registered uint16 millimetres HxW");
    image_ = cv::Mat(intrinsics_.height, intrinsics_.width, CV_16UC1, a.mutable_data()).clone();
  }
};

class SingleCameraTracker {
 public:
  SingleCameraTracker(const std::string& mesh, const std::string& cache,
                      const Eigen::Matrix3f& k, ColorArray color, DepthArray depth,
                      const Eigen::Matrix4f& initial, int threads) {
    omp_set_num_threads(threads);
    m3t::Intrinsics intr{k(0,0), k(1,1), k(0,2), k(1,2), int(color.shape(1)), int(color.shape(0))};
    color_ = std::make_shared<MemoryColor>(intr);
    depth_ = std::make_shared<MemoryDepth>(intr);
    color_->put(color); depth_->put(depth);
    body_ = std::make_shared<m3t::Body>("object", mesh, 1.0f, true, false, m3t::Transform3fA::Identity());
    auto region_model = std::make_shared<m3t::RegionModel>("region_model", body_, std::filesystem::path(cache) / "region.bin");
    auto depth_model = std::make_shared<m3t::DepthModel>("depth_model", body_, std::filesystem::path(cache) / "depth.bin");
    auto region = std::make_shared<m3t::RegionModality>("region", body_, color_, region_model);
    auto depth_mod = std::make_shared<m3t::DepthModality>("depth", body_, depth_, depth_model);
    region->MeasureOcclusions(depth_);
    depth_mod->MeasureOcclusions();
    auto link = std::make_shared<m3t::Link>("object_link", body_);
    require(link->AddModality(region), "add region");
    require(link->AddModality(depth_mod), "add depth");
    tracker_ = std::make_shared<m3t::Tracker>("single_camera_tracker");
    require(tracker_->AddOptimizer(std::make_shared<m3t::Optimizer>("optimizer", link)), "add optimizer");
    {
      ModelLog log(std::filesystem::path(cache) / "model_setup.log");
      require(tracker_->SetUp(), "setup/model generation (see cache/model_setup.log)");
    }
    body_->set_body2world_pose(m3t::Transform3fA(initial));
    require(tracker_->CalculateConsistentPoses(), "initial link pose");
    tracker_->StartTracking();
    require(tracker_->ExecuteStartingStep(0), "initial histograms");
    geometry_ = std::make_shared<m3t::RendererGeometry>("quality_geometry");
    require(geometry_->AddBody(body_), "quality body");
    require(geometry_->SetUp(), "quality geometry");
    renderer_ = std::make_shared<m3t::FullBasicDepthRenderer>("quality", geometry_, color_);
    require(renderer_->SetUp(), "quality renderer");
  }
  Eigen::Matrix4f track(ColorArray color, DepthArray depth) {
    color_->put(color); depth_->put(depth);
    require(tracker_->ExecuteTrackingStep(++iteration_), "tracking step");
    Eigen::Matrix4f result = body_->body2world_pose().matrix();
    if (!result.allFinite()) throw std::runtime_error("Non-finite M3T pose");
    return result;
  }
  py::array_t<float> rendered_depth() {
    require(renderer_->StartRendering(), "render quality depth");
    require(renderer_->FetchDepthImage(), "fetch quality depth");
    const auto& image = renderer_->depth_image();
    py::array_t<float> result({image.rows, image.cols});
    auto dst = result.mutable_unchecked<2>();
    for (int v=0; v<image.rows; ++v)
      for (int u=0; u<image.cols; ++u) {
        auto value = image.at<unsigned short>(v,u);
        dst(v,u) = value == std::numeric_limits<unsigned short>::max() ? 0.0f : renderer_->Depth(value);
      }
    return result;
  }
 private:
  int iteration_ = 0;
  std::shared_ptr<MemoryColor> color_;
  std::shared_ptr<MemoryDepth> depth_;
  std::shared_ptr<m3t::Body> body_;
  std::shared_ptr<m3t::Tracker> tracker_;
  std::shared_ptr<m3t::RendererGeometry> geometry_;
  std::shared_ptr<m3t::FullBasicDepthRenderer> renderer_;
};

PYBIND11_MODULE(_m3t_native, m) {
  py::class_<SingleCameraTracker>(m, "SingleCameraTracker")
    .def(py::init<const std::string&, const std::string&, const Eigen::Matrix3f&,
         ColorArray, DepthArray, const Eigen::Matrix4f&, int>())
    .def("track", &SingleCameraTracker::track)
    .def("rendered_depth", &SingleCameraTracker::rendered_depth);
}
