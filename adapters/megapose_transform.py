"""NumPy/SciPy implementation of MegaPose's inference-only SE(3) interface.

Used when Pinocchio is unavailable on Windows. Quaternion order is xyzw.
This does not implement robotics/dynamics APIs or change pose estimation.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class Quaternion:
    def __init__(self, matrix):
        self._rotation = Rotation.from_matrix(matrix)

    def coeffs(self):
        return self._rotation.as_quat()

    def matrix(self):
        return self._rotation.as_matrix()


class Transform:
    def __init__(self, *args):
        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)
        if len(args) == 1:
            self._matrix = array(args[0]).copy()
            if self._matrix.shape != (4, 4):
                raise ValueError("Transform requires a 4x4 matrix")
        elif len(args) == 2:
            rotation, translation = args
            rotation = rotation.matrix() if isinstance(rotation, Quaternion) else array(rotation)
            if rotation.size == 4:
                rotation = Rotation.from_quat(rotation.reshape(4)).as_matrix()
            if rotation.shape != (3, 3):
                raise ValueError("Rotation requires xyzw quaternion or 3x3 matrix")
            self._matrix = np.eye(4)
            self._matrix[:3, :3] = rotation
            self._matrix[:3, 3] = array(translation).reshape(3)
        else:
            raise ValueError("Transform expects matrix or rotation, translation")

    def __mul__(self, other):
        return Transform(self.matrix @ other.matrix)

    def inverse(self):
        result = np.eye(4)
        result[:3, :3] = self.matrix[:3, :3].T
        result[:3, 3] = -result[:3, :3] @ self.translation
        return Transform(result)

    def toHomogeneousMatrix(self):
        return self.matrix

    @property
    def matrix(self):
        return self._matrix

    @property
    def translation(self):
        return self.matrix[:3, 3].copy()

    @property
    def quaternion(self):
        return Quaternion(self.matrix[:3, :3])
