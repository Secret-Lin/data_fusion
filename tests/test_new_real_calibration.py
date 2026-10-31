import cv2
import numpy as np

from tools.calibrate_new_real_sessions import project_error


def test_heldout_projection_detects_bad_transform_without_pnp():
    points=np.array([[-.02,-.02,0],[.02,-.02,0],[.02,.02,0],[-.02,.02,0]])
    k=np.array([[500.,0,200],[0,500.,200],[0,0,1]])
    pose=np.eye(4);pose[2,3]=-1.
    pixels=cv2.projectPoints(points,np.zeros(3),np.array([0.,0.,1.]),k,None)[0].reshape(-1,2)
    assert project_error(pose,points,pixels,k,np.zeros(5))<1e-10
    wrong=pose.copy();wrong[0,3]=.1
    assert project_error(wrong,points,pixels,k,np.zeros(5))>40
    behind=pose.copy();behind[2,3]=1.
    assert project_error(behind,points,pixels,k,np.zeros(5)) is None
