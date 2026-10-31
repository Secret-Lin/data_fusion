from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CnosMask:
    object_name: str
    mask: np.ndarray
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    score_margin: float = 1.0


class CnosRunner:
    """Small runtime wrapper around the public CNOS CAD matching pipeline.

    CNOS proposes masks with SAM and identifies each proposal by comparing
    DINOv2 descriptors against cached renderings of the configured HOPE CAD
    models. Heavy dependencies are imported lazily so the normal backend does
    not require them unless recovery is enabled.
    """

    _MODEL_TO_CHECKPOINT = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }

    def __init__(
        self,
        *,
        cnos_root: str | Path,
        template_dir: str | Path,
        sam_model_type: str,
        sam_checkpoint: str | Path,
        dino_model_name: str,
        dino_root: str | Path,
        dino_checkpoint: str | Path,
        device: str = "cuda",
        image_width: int = 640,
        confidence_threshold: float = 0.45,
        min_score_margin: float = 0.03,
        min_mask_fraction: float = 3e-4,
        max_proposals: int = 100,
    ) -> None:
        self.cnos_root = Path(cnos_root)
        self.template_dir = Path(template_dir)
        self.sam_model_type = str(sam_model_type)
        self.sam_checkpoint = Path(sam_checkpoint)
        self.dino_model_name = str(dino_model_name)
        self.dino_root = Path(dino_root)
        self.dino_checkpoint = Path(dino_checkpoint)
        self.device_name = str(device)
        self.image_width = max(0, int(image_width))
        self.confidence_threshold = float(confidence_threshold)
        self.min_score_margin = max(0.0, float(min_score_margin))
        self.min_mask_fraction = max(0.0, float(min_mask_fraction))
        self.max_proposals = max(1, int(max_proposals))

        self._lock = threading.RLock()
        self._loaded = False
        self._objects: list[str] = []
        self._template_features = None
        self._sam_generator = None
        self._dino = None
        self._torch = None
        self._reference_features: dict[str, object] = {}
        self.last_candidates: list[dict[str, object]] = []

    def prepare(self, object_names: list[str]) -> None:
        with self._lock:
            requested = [str(name) for name in object_names]
            if self._loaded and requested == self._objects:
                return
            self._load_models_and_templates(requested)

    def detect(
        self,
        color_bgr: np.ndarray,
        object_names: list[str] | None = None,
    ) -> list[CnosMask]:
        with self._lock:
            if not self._loaded:
                self.prepare(object_names or self._objects)
            requested = set(object_names or self._objects)
            if not requested:
                return []

            bgr = np.asarray(color_bgr)
            if bgr.ndim != 3 or bgr.shape[2] < 3:
                raise ValueError("CNOS color_bgr must have shape HxWx3")
            rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
            original_height, original_width = rgb.shape[:2]
            scale = 1.0
            inference_rgb = rgb
            if self.image_width and original_width != self.image_width:
                scale = self.image_width / float(original_width)
                inference_rgb = cv2.resize(
                    rgb,
                    (
                        self.image_width,
                        max(1, int(round(original_height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            proposals = self._sam_generator.generate(inference_rgb)
            proposals = self._filter_proposals(proposals, inference_rgb.shape[:2])
            if not proposals:
                return []

            proposal_images = [
                self._masked_crop(inference_rgb, item["segmentation"])
                for item in proposals
            ]
            query_features = self._compute_features(proposal_images)
            torch = self._torch
            similarities = torch.einsum(
                "qd,otd->qot", query_features, self._template_features
            )
            top_count = min(5, similarities.shape[-1])
            scores_per_object = torch.topk(
                similarities, k=top_count, dim=-1
            ).values.mean(dim=-1)
            for object_index, object_name in enumerate(self._objects):
                reference_features = self._reference_features.get(object_name)
                if reference_features is None:
                    continue
                reference_scores = torch.einsum(
                    "qd,rd->qr", query_features, reference_features
                ).max(dim=-1).values
                scores_per_object[:, object_index] = (
                    0.65 * scores_per_object[:, object_index]
                    + 0.35 * reference_scores
                )
            proposal_scores, object_indices = scores_per_object.max(dim=1)

            self.last_candidates = []
            for proposal, raw_score, raw_object_index, raw_score_distribution in zip(
                proposals,
                proposal_scores.tolist(),
                object_indices.tolist(),
                scores_per_object.tolist(),
            ):
                object_name = self._objects[int(raw_object_index)]
                score = float(raw_score)
                other_scores = [
                    float(value)
                    for index, value in enumerate(raw_score_distribution)
                    if index != int(raw_object_index)
                ]
                score_margin = score - max(other_scores) if other_scores else 1.0
                proposal_mask = np.asarray(proposal["segmentation"], dtype=np.uint8)
                proposal_ys, proposal_xs = np.nonzero(proposal_mask)
                if len(proposal_xs):
                    self.last_candidates.append(
                        {
                            "object_name": object_name,
                            "score": score,
                            "score_margin": score_margin,
                            "area": int(len(proposal_xs)),
                            "bbox_xyxy": (
                                int(proposal_xs.min()),
                                int(proposal_ys.min()),
                                int(proposal_xs.max()),
                                int(proposal_ys.max()),
                            ),
                            "scores": {
                                name: float(value)
                                for name, value in zip(
                                    self._objects, raw_score_distribution
                                )
                            },
                        }
                    )
            # The runtime tracks one instance of each configured HOPE object.
            # Assign proposals jointly so a visually similar can cannot claim
            # the best masks of multiple known objects independently.
            from scipy.optimize import linear_sum_assignment

            score_matrix = scores_per_object.detach().cpu().numpy().T
            object_rows, proposal_columns = linear_sum_assignment(
                -score_matrix
            )
            selected: dict[str, CnosMask] = {}
            for object_index, proposal_index in zip(
                object_rows.tolist(), proposal_columns.tolist()
            ):
                object_name = self._objects[int(object_index)]
                score = float(score_matrix[object_index, proposal_index])
                competing_scores = np.delete(
                    score_matrix[:, proposal_index], object_index
                )
                score_margin = (
                    score - float(np.max(competing_scores))
                    if competing_scores.size
                    else 1.0
                )
                if (
                    object_name not in requested
                    or score < self.confidence_threshold
                    or score_margin < self.min_score_margin
                ):
                    continue
                proposal = proposals[int(proposal_index)]
                mask_small = np.asarray(proposal["segmentation"], dtype=np.uint8)
                if scale != 1.0:
                    mask = cv2.resize(
                        mask_small,
                        (original_width, original_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    mask = mask_small
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    continue
                candidate = CnosMask(
                    object_name=object_name,
                    mask=(mask > 0).astype(np.uint8) * 255,
                    score=score,
                    bbox_xyxy=(
                        int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    ),
                    score_margin=score_margin,
                )
                selected[object_name] = candidate

            # A proposal can have nested SAM variants. If two selected masks
            # still describe the same image region, keep only the more
            # confident identity instead of re-registering two CAD models on it.
            names = list(selected)
            rejected: set[str] = set()
            for first_index, first_name in enumerate(names):
                first_mask = selected[first_name].mask > 0
                for second_name in names[first_index + 1:]:
                    second_mask = selected[second_name].mask > 0
                    intersection = int(np.count_nonzero(first_mask & second_mask))
                    first_area = int(np.count_nonzero(first_mask))
                    second_area = int(np.count_nonzero(second_mask))
                    union = first_area + second_area - intersection
                    overlap_smaller = intersection / max(1, min(first_area, second_area))
                    overlap_union = intersection / max(1, union)
                    if overlap_union <= 0.35 and overlap_smaller <= 0.50:
                        continue
                    first_score = selected[first_name].score
                    second_score = selected[second_name].score
                    rejected.add(
                        second_name if first_score >= second_score else first_name
                    )
            return [
                selected[name]
                for name in self._objects
                if name in selected and name not in rejected
            ]

    def add_reference(
        self,
        color_bgr: np.ndarray,
        object_name: str,
        mask: np.ndarray,
    ) -> None:
        """Add a verified real-image view to disambiguate similar HOPE cans."""
        with self._lock:
            if not self._loaded:
                raise RuntimeError("CNOS must be prepared before adding references")
            resolved = next(
                (name for name in self._objects if name.casefold() == object_name.casefold()),
                None,
            )
            if resolved is None:
                raise KeyError(f"Unknown CNOS reference object: {object_name}")
            bgr = np.asarray(color_bgr)
            binary = np.asarray(mask) > 0
            if bgr.shape[:2] != binary.shape:
                raise ValueError("CNOS reference mask and color resolutions must match")
            rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
            feature = self._compute_features([self._masked_crop(rgb, binary)])
            previous = self._reference_features.get(resolved)
            if previous is None:
                self._reference_features[resolved] = feature
            else:
                self._reference_features[resolved] = self._torch.cat(
                    (previous, feature), dim=0
                )[-8:]

    def _load_models_and_templates(self, object_names: list[str]) -> None:
        if not object_names:
            raise ValueError("CNOS recovery requires at least one object")
        if self.sam_model_type not in self._MODEL_TO_CHECKPOINT:
            raise ValueError(
                f"Unsupported SAM model {self.sam_model_type!r}; expected "
                f"{sorted(self._MODEL_TO_CHECKPOINT)}"
            )
        if not self.cnos_root.is_dir():
            raise FileNotFoundError(f"CNOS source directory not found: {self.cnos_root}")
        if not self.sam_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.sam_checkpoint}; "
                "download the checkpoint before enabling CNOS recovery"
            )
        if not self.dino_root.is_dir():
            raise FileNotFoundError(
                f"DINOv2 source directory not found: {self.dino_root}"
            )
        if not self.dino_checkpoint.is_file():
            raise FileNotFoundError(
                f"DINOv2 checkpoint not found: {self.dino_checkpoint}; "
                "download it before enabling CNOS recovery"
            )

        root_text = str(self.cnos_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CNOS recovery requested CUDA but CUDA is unavailable")
        device = torch.device(self.device_name)
        sam = sam_model_registry[self.sam_model_type](
            checkpoint=str(self.sam_checkpoint)
        ).to(device=device)
        sam.eval()
        self._sam_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            box_nms_thresh=0.7,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )
        dino_root_text = str(self.dino_root.resolve())
        if dino_root_text not in sys.path:
            sys.path.insert(0, dino_root_text)
        from dinov2.hub import backbones as dino_backbones

        dino_builder = getattr(dino_backbones, self.dino_model_name, None)
        if dino_builder is None or not self.dino_model_name.startswith("dinov2_vit"):
            raise ValueError(f"Unsupported local DINOv2 model: {self.dino_model_name}")
        self._dino = dino_builder(pretrained=False)
        try:
            state_dict = torch.load(
                self.dino_checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(self.dino_checkpoint, map_location="cpu")
        self._dino.load_state_dict(state_dict, strict=True)
        self._dino = self._dino.to(device=device)
        self._dino.eval()
        self._torch = torch
        self._objects = list(object_names)
        self._reference_features.clear()

        features = []
        for object_name in self._objects:
            object_dir = self.template_dir / object_name
            paths = sorted(object_dir.glob("*.png"))
            if not paths:
                raise FileNotFoundError(
                    f"No CNOS template PNG files for {object_name}: {object_dir}"
                )
            images = []
            for path in paths:
                rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if rgba is None:
                    continue
                if rgba.ndim == 3 and rgba.shape[2] == 4:
                    mask = rgba[..., 3] > 0
                    template_rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
                else:
                    template_rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
                    mask = np.any(template_rgb > 0, axis=2)
                images.append(self._masked_crop(template_rgb, mask))
            if not images:
                raise RuntimeError(f"All CNOS templates are unreadable: {object_dir}")
            features.append(self._compute_features(images))
        self._template_features = torch.stack(features, dim=0)
        self._loaded = True

    def _filter_proposals(self, proposals: list[dict], shape: tuple[int, int]) -> list[dict]:
        height, width = shape
        min_pixels = self.min_mask_fraction * float(height * width)
        kept = []
        for proposal in proposals:
            mask = np.asarray(proposal.get("segmentation"), dtype=bool)
            area = int(proposal.get("area", int(mask.sum())))
            if mask.shape != (height, width) or area < min_pixels:
                continue
            if area > 0.65 * height * width:
                continue
            kept.append(proposal)
        kept.sort(
            key=lambda item: (
                float(item.get("stability_score", 0.0)),
                float(item.get("predicted_iou", 0.0)),
            ),
            reverse=True,
        )
        return kept[: self.max_proposals]

    def _masked_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            raise ValueError("CNOS proposal/template mask is empty")
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = np.asarray(rgb[y1:y2, x1:x2, :3]).copy()
        crop_mask = binary[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        size = max(crop.shape[:2])
        square = np.zeros((size, size, 3), dtype=np.uint8)
        y = (size - crop.shape[0]) // 2
        x = (size - crop.shape[1]) // 2
        square[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        return cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)

    def _compute_features(self, images: list[np.ndarray]):
        torch = self._torch
        device = next(self._dino.parameters()).device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        output = []
        with torch.inference_mode():
            for offset in range(0, len(images), 16):
                batch_np = np.stack(images[offset:offset + 16])
                batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                batch = batch.permute(0, 3, 1, 2) / 255.0
                batch = (batch - mean) / std
                feature = self._dino(batch)
                output.append(torch.nn.functional.normalize(feature, dim=-1))
        return torch.cat(output, dim=0)
