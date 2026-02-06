import numpy as np
from mathutils import *

def dlt_homography(
    obj_xy: np.ndarray, 
    quad_xy: np.ndarray, 
    eps: float=1e-12
) -> np.ndarray:
    """
    Inputs
    - obj_xy: (4,2) real corner points of the square
    - quad_xy: (4,2) distorted corner points of the square in the image

    Return
    - H_mat: (3,3) Homography matrix
    """
    T_obj, obj_n = normalize_quadpoints(obj_xy, eps)
    T_img, img_n = normalize_quadpoints(quad_xy, eps)

    # formulate matrix of real quad and image quad correspondance
    A = []
    for (X, Y), (u, v) in zip(obj_n, img_n):
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
    A = np.asarray(A, dtype=np.float64)

    # solve using single value decomposition
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] # flattened homography
    H_norm = h.reshape(3, 3)

    H_mat = np.linalg.inv(T_img) @ H_norm @ T_obj
    H_mat /= (H_mat[2, 2] + eps)
    return H_mat


def pos_from_quad(
    quad_xy: np.ndarray,
    K: np.ndarray,
    len_qr: float,
    eps: float = 1e-12
) -> QRtransform:
    """
    Estimate orientation and position of a square plane in camera coordinate

    Inputs
    - quad_xy: (4,2) quad points in the image pixels
    - K: (3,3) camera intrinsic matrix
    - len_qr: side length of qr square in meters

    Returns
    - R: (3,3) rotation matrix (SO(3)), mapping square frame -> camera frame
    - t: (3,) translation vector in camera coords (same units as len_qr)
    """

    obj_xy = np.array([
        [-len_qr / 2.0, -len_qr / 2.0],
        [-len_qr / 2.0,  len_qr / 2.0],
        [ len_qr / 2.0,  len_qr / 2.0],
        [ len_qr / 2.0, -len_qr / 2.0],
    ], dtype=np.float64)

    H = dlt_homography(obj_xy, quad_xy)

    # Decompose: H = K [r1 r2 t]  (for Z=0 plane coords [X,Y,1])
    K_inv = np.linalg.inv(K)
    B = K_inv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # calculate scaling lambda. use average of 2 columns for stability
    lam = 0.5 * (1.0 / (np.linalg.norm(b1) + eps) + 1.0 / (np.linalg.norm(b2) + eps))

    r1 = lam * b1
    r2 = lam * b2
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack([r1, r2, r3])

    # Project to nearest rotation matrix (SVD)
    U, _, Vt = np.linalg.svd(R_approx)
    R = np.matmul(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.matmul(U, Vt)

    t = lam * b3

    # enforce positive Z to meet "in front of camera" condition
    if t[2] < 0:
        R[:, 0] *= -1
        R[:, 1] *= -1
        t *= -1

    return QRtransform(R, t)


def reorder_quad(quad_xy: np.ndarray, zbar_orientation) -> np.ndarray:
    """
    Inputs:
    - quad_xy: (4,2) in [TL, TR, BR, BL] order.
    - orientation info given from the python zbar library

    Returns 
    - reordered quad so that indices correspond to the QR's logical corners.
    """
    ori = zbar_orientation.upper()
    mapping = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
    k = mapping[ori]

    return np.roll(quad_xy, shift=-k, axis=0)

