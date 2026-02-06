import numpy as np
from qrdata import QRtransform

def normalize_quadpoints(points: np.ndarray, eps:float = 1e-12):
    """
    normalize quad points (should be on the same plane)

    Inputs
    - points: 4 corner poisitions of the quad
    - eps: epsilon for stability
    
    Returns
    - normalization matrix
    - normalized quad points
    """
    center = points.mean(axis=0)
    dist = np.linalg.norm(points - center, axis=1).mean()
    s = np.sqrt(2.0) / (dist + eps)
    T = np.array([[s, 0, -s * center[0]],
                  [0, s, -s * center[1]],
                  [0, 0, 1]], dtype=np.float64)
    
    points_h = np.c_[points, np.ones((points.shape[0], 1), dtype=np.float64)]
    points_n = np.matmul(T, points_h.T).T

    return T, points_n[:, :2]

def angle_from_transform(T: QRtransform):
    R_mat = T.rot
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return theta

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