from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from adapters.quest_init import QuestTagInitializer


def initializer(tmp_path):
    path=tmp_path/'cube.yaml'
    path.write_text(yaml.safe_dump({'dictionary':'DICT_4X4_50','tag_size_m':.04,
                                  'tags':{0:{'T_world_tag':np.eye(4).tolist()}}}))
    return QuestTagInitializer(path,min_samples=3,strict_geometry=True)


@pytest.mark.parametrize('depth,rms', [(-1.,0.),(1.,500.)])
def test_rejects_behind_camera_or_large_reprojection(tmp_path,monkeypatch,depth,rms):
    qi=initializer(tmp_path)
    qi.detector=SimpleNamespace(detectMarkers=lambda im:([np.array([[[190,190],[210,190],[210,210],[190,210]]])],np.array([[0]]),[]))
    bad=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,depth]),
                        quality=SimpleNamespace(rms_px=rms,max_px=rms))
    monkeypatch.setattr(qi._geometry_solver,'_solve_single_tag',lambda *args:bad)
    qi.add_frame(np.zeros((400,400,3),np.uint8),{'intrinsics':{'fx':500.,'fy':500.,'cx':200.,'cy':200.},
                 'cameraPose':{'position':[0,0,0],'rotation':[0,0,0,1]}})
    assert not qi.samples and qi.geometry_rejections==1


def test_filtered_outlier_never_reenters_optimization(tmp_path):
    qi=initializer(tmp_path)
    good=np.eye(4);good[2,3]=-1.
    bad=good.copy();bad[0,3]=100.
    k=np.array([[500.,0,200.],[0,500.,200.],[0,0,1.]])
    points=qi._world_corners(0)
    pixels=cv2.projectPoints(points,np.zeros(3),np.array([0.,0.,1.]),k,None)[0].reshape(-1,2)
    qi.samples=[good.copy() for _ in range(3)]+[bad]
    qi.observations=[(points,pixels,k,np.eye(4)) for _ in range(3)]+[(points,pixels+5000,k,np.eye(4))]
    result=qi.finish()
    np.testing.assert_allclose(result,good,atol=1e-6)
    assert qi.fit_report['fit_frames']==3
    assert qi.fit_report['pose_outlier_frames']==1
