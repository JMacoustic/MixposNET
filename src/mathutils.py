import numpy as np

class Orientation:
    def __init__(self, rotation: np.ndarray = np.identity(3), translation: np.ndarray = np.zeros((3, 1))):
        self.rot = rotation
        self.trans = translation
    
    def reset(self):
        self.rot = np.identity(3)
        self.trans = np.zeros((3, 1))

def inverse_transform(T: Orientation):
    R_mat = T.rot
    P_vec = T.trans

    R_inv = R_mat.T
    P_inv = - np.matmul(R_inv, P_vec)

    return Orientation(R_inv, P_inv)


def multiple_transform(T1: Orientation, T2: Orientation):
    R_mat1 = T1.rot
    P_vec1 = T1.trans
    R_mat2 = T2.rot
    P_vec2 = T2.trans

    R_mul = np.matmul(R_mat1, R_mat2)
    P_mul = np.matmul(R_mat1, P_vec2) + P_vec1

    return Orientation(R_mul, P_mul)

def angle_from_transform(T: Orientation):
    R_mat = T.rot
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return theta

def get_21_transform(transform_c1: Orientation, transform_c2: Orientation) -> Orientation:
    """Input 2 SE3 transforms that shares same reference frame. Returns relative SE3 transform between them"""
    T_2c = inverse_transform(transform_c2)
    T_c1 = transform_c1

    transform_21 = multiple_transform(T_2c, T_c1)

    return transform_21

def get_camera_intrinsic(
    fx: float, fy: float,
    cx: float, cy: float
) -> np.ndarray:
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K
