import numpy as np
from model.qreader_moon import QReader
import cv2
import numpy as np
from qrtransforms import *

def _dlt_homography(obj_xy: np.ndarray, img_uv: np.ndarray) -> np.ndarray:
    """
    Compute homography H (3x3) such that  s*[u,v,1]^T = H*[X,Y,1]^T
    using normalized DLT. Numpy-only.
    obj_xy: (4,2) plane points
    img_uv: (4,2) image points
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    assert obj_xy.shape == (4, 2) and img_uv.shape == (4, 2)

    def normalize_2d(pts):
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1).mean()
        s = np.sqrt(2.0) / (d + 1e-12)
        T = np.array([[s, 0, -s * c[0]],
                      [0, s, -s * c[1]],
                      [0, 0, 1]], dtype=np.float64)
        pts_h = np.c_[pts, np.ones((pts.shape[0], 1), dtype=np.float64)]
        pts_n = (T @ pts_h.T).T
        return T, pts_n[:, :2]

    T_obj, obj_n = normalize_2d(obj_xy)
    T_img, img_n = normalize_2d(img_uv)

    A = []
    for (X, Y), (u, v) in zip(obj_n, img_n):
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3, 3)

    # denormalize: img ~ T_img^{-1} * Hn * T_obj * obj
    H = np.linalg.inv(T_img) @ Hn @ T_obj
    H /= (H[2, 2] + 1e-12)
    return H


def square_pose_from_4pts(
    img_pts_uv: np.ndarray,
    K: np.ndarray,
    side_len: float,
) -> QRtransform:
    """
    Estimate pose of a square planar target (Z=0 in its own frame) in camera coords.

    Inputs
    - img_pts_uv: (4,2) image points in pixels, consistent order with obj points below
                 Order assumed:
                 0: (-s/2, -s/2), 1: (-s/2, +s/2), 2: (+s/2, +s/2), 3: (+s/2, -s/2)
                 (i.e., CCW around the square, starting at bottom-left in square coords)
    - K: (3,3) camera intrinsic matrix
    - side_len: square side length in meters (or any length unit)

    Returns
    - R: (3,3) rotation matrix (SO(3)), mapping square frame -> camera frame
    - t: (3,) translation vector in camera coords (same units as side_len)

    Notes
    - Assumes pinhole camera model and zero distortion.
    - If you have distortion, undistort the image points first.
    """
    img_pts_uv = np.asarray(img_pts_uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    assert img_pts_uv.shape == (4, 2)
    assert K.shape == (3, 3)

    s = float(side_len)
    obj_xy = np.array([
        [-s / 2.0, -s / 2.0],
        [-s / 2.0,  s / 2.0],
        [ s / 2.0,  s / 2.0],
        [ s / 2.0, -s / 2.0],
    ], dtype=np.float64)

    H = _dlt_homography(obj_xy, img_pts_uv)

    # Decompose: H = K [r1 r2 t]  (for Z=0 plane coords [X,Y,1])
    Kinv = np.linalg.inv(K)
    B = Kinv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # scale (use both columns for stability)
    lam1 = 1.0 / (np.linalg.norm(b1) + 1e-12)
    lam2 = 1.0 / (np.linalg.norm(b2) + 1e-12)
    lam = 0.5 * (lam1 + lam2)

    r1 = lam * b1
    r2 = lam * b2
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack([r1, r2, r3])

    # Project to nearest rotation matrix (SVD)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = lam * b3  # (3,)

    # "in front of camera" convention: enforce positive Z translation
    if t[2] < 0:
        R[:, 0] *= -1
        R[:, 1] *= -1
        R[:, 2] *= 1  # keeps det=+1 after the above two flips
        t *= -1

    return QRtransform(R, t)

def _ori_to_k(ori) -> int:
    # returns k in {0,1,2,3} meaning rotate corners by k*90° CW
    if ori is None:
        return 0
    if isinstance(ori, bytes):
        ori = ori.decode("utf-8", errors="ignore")
    if isinstance(ori, str):
        ori = ori.upper()
        # These are pyzbar common values
        mapping = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
        if ori in mapping:
            return mapping[ori]
        # sometimes it might be "0", "90", ...
        try:
            deg = int(ori)
            return (deg // 90) % 4
        except Exception:
            return 0
    if isinstance(ori, (int, np.integer)):
        # could be degrees or already 0..3 depending on platform
        if ori in (0, 1, 2, 3):
            return int(ori)
        return (int(ori) // 90) % 4
    return 0

def reorder_quad_by_zbar_orientation(quad_xy: np.ndarray, zbar_orientation) -> np.ndarray:
    """
    quad_xy: (4,2) in [TL, TR, BR, BL] order.
    Returns quad reordered so that indices correspond to the QR's logical corners.
    """
    quad_xy = np.asarray(quad_xy, dtype=np.float64)
    k = _ori_to_k(zbar_orientation)  # 0..3, CW steps
    # If QR is rotated CW by k, then to map to logical corners, rotate corner list CCW by k
    # (equivalently roll by -k)
    return np.roll(quad_xy, shift=-k, axis=0)

def transform_to_angle(T):
    R_mat = T.rot
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])  # radians, [-pi, pi]
    return theta