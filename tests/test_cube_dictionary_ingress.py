from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import yaml

from adapters.quest_init import QuestTagInitializer
from common.replay_extrinsics import ensure_replay_extrinsics
from common.tag_dictionary import cube_dictionary
from standalone.tag_cube_localizer_replay_with_cube_model import load_cube_model
from tools.prepare_generator_cube_model import convert_generator_model

SOURCE = Path(__file__).resolve().parents[1] / "standalone/cube_model.yaml"


@pytest.mark.parametrize("name", ["DICT_APRILTAG_16h5", "DICT_4X4_50"])
def test_dictionary_replay_generation(tmp_path, name):
    import json
    data=yaml.safe_load(SOURCE.read_text(encoding="utf-8"));data['dictionary']=name
    model=tmp_path/'cube.yaml';model.write_text(yaml.safe_dump(data),encoding='utf-8')
    assert load_cube_model(model).dictionary==name
    marker=cv2.aruco.generateImageMarker(cube_dictionary(name),0,160)
    im=np.full((400,400,3),255,np.uint8);im[120:280,120:280]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
    (tmp_path/'rgbd').mkdir()
    with h5py.File(tmp_path/'rgbd/cam.h5','w') as f:
        f.create_dataset('color',data=np.repeat(im[None],3,axis=0));f.attrs['color_format']='bgr8'
    (tmp_path/'camera_metadata.json').write_text(json.dumps({'cam':{'K':[[800,0,200],[0,800,200],[0,0,1]],'dist':[0]*5}}))
    result=ensure_replay_extrinsics(tmp_path,cube_model_path=model)
    assert 'cam' in result.fixed


def test_unknown_dictionary_rejected(tmp_path):
    data=yaml.safe_load(SOURCE.read_text(encoding='utf-8'));data['dictionary']='NOT_A_DICTIONARY'
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(data),encoding='utf-8')
    with pytest.raises(ValueError,match='Unsupported cube dictionary'):load_cube_model(path)
    with pytest.raises(ValueError,match='Unsupported cube dictionary'):QuestTagInitializer(path)


def test_generator_conversion_preserves_physical_corner_projection(tmp_path):
    data=yaml.safe_load(SOURCE.read_text(encoding='utf-8'))
    output=convert_generator_model(data)
    path=tmp_path/'down.yaml';path.write_text(yaml.safe_dump(output),encoding='utf-8')
    loaded=load_cube_model(path);basis=np.diag([1.,-1.,-1.,1.])
    h=data['tag_size_m']/2
    local=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    pose=np.eye(4);pose[:3,:3],_=cv2.Rodrigues(np.array([.1,.2,-.1]));pose[2,3]=.5
    k=np.array([[800.,0,320],[0,800,240],[0,0,1]])
    for tid,entry in data['tags'].items():
        native=(np.array(entry['T_world_tag'])@np.c_[local,np.ones(4)].T).T[:,:3]
        down=loaded.world_tag_corners(tid)
        assert np.allclose(down,(basis@np.c_[native,np.ones(4)].T).T[:,:3])
        rv,_=cv2.Rodrigues(pose[:3,:3]);rd,_=cv2.Rodrigues((pose@basis)[:3,:3])
        uv,_=cv2.projectPoints(native,rv,pose[:3,3],k,None)
        ud,_=cv2.projectPoints(down,rd,pose[:3,3],k,None)
        assert np.allclose(uv,ud,atol=1e-8)
    with pytest.raises(ValueError):convert_generator_model(output)


def test_ambiguous_generator_axes_require_explicit_preparation(tmp_path):
    data=yaml.safe_load(SOURCE.read_text(encoding='utf-8'))
    data.pop('coordinate_system',None);data['dictionary']='DICT_4X4_50'
    path=tmp_path/'native.yaml';path.write_text(yaml.safe_dump(data),encoding='utf-8')
    with pytest.raises(ValueError,match='prepare a separate Y-down'):
        load_cube_model(path)
    with pytest.raises(ValueError,match='prepare a separate Y-down'):
        QuestTagInitializer(path)
