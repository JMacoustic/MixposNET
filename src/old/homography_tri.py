import numpy as np
from qrdata import QRtransform
from mathutils import normalize_points

def dlt_affine(
    obj_xy: np.ndarray,
    tri_xy: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Estimate affine transform H (3x3, last row [0,0,1]) from 3 point correspondences.

    H maps [X,Y,1]^T (object plane) -> [u,v,1]^T (image plane) under affine model:
      u = aX + bY + c
      v = dX + eY + f
    """
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(-1, 2)
    img = np.asarray(tri_xy, dtype=np.float64).reshape(-1, 2)

    if obj.shape[0] != 3 or img.shape[0] != 3:
        raise ValueError(f"dlt_affine expects exactly 3 points. got obj={obj.shape}, img={img.shape}")

    T_obj, obj_n = normalize_points(obj, eps)
    T_img, img_n = normalize_points(img, eps)

    # Solve for affine params on normalized points:
    # [X Y 1 0 0 0] [a b c]^T = u
    # [0 0 0 X Y 1] [d e f]^T = v
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros((6,), dtype=np.float64)

    for i, ((X, Y), (u, v)) in enumerate(zip(obj_n, img_n)):
        A[2*i + 0, :] = [X, Y, 1.0, 0.0, 0.0, 0.0]
        A[2*i + 1, :] = [0.0, 0.0, 0.0, X, Y, 1.0]
        b[2*i + 0] = u
        b[2*i + 1] = v

    # Direct solve (3 pts => square system). Use lstsq as a safety net.
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)

    a, b1, c, d, e, f = x.tolist()

    H_norm = np.array([[a,  b1, c],
                       [d,  e,  f],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

    # Denormalize: x_img = inv(T_img) * H_norm * T_obj * x_obj
    H = np.linalg.inv(T_img) @ H_norm @ T_obj
    H /= (H[2, 2] + eps)
    return H


def pos_from_triangle(
    tri_xy: np.ndarray,
    K: np.ndarray,
    len_qr: float,
    eps: float = 1e-12
) -> QRtransform:
    """
    Pose estimate from 3 image points using an AFFINE planar mapping.
    This is an approximation vs full homography (needs 4 points).

    Assumes the 3 points correspond to 3 known object-plane points.
    Here we use three corners of the square in this fixed order:
      P0 = (-L/2, -L/2)
      P1 = (-L/2, +L/2)
      P2 = (+L/2, +L/2)

    If your triangle points are a different subset/order, you MUST match it.
    """
    tri = np.asarray(tri_xy, dtype=np.float64).reshape(-1, 2)
    if tri.shape[0] != 3:
        raise ValueError(f"pos_from_triangle expects (3,2). got {tri.shape}")

    obj_xy = np.array([
        [-len_qr / 2.0, -len_qr / 2.0],
        [-len_qr / 2.0,  len_qr / 2.0],
        [ len_qr / 2.0,  len_qr / 2.0],
    ], dtype=np.float64)

    H = dlt_affine(obj_xy, tri, eps=eps)

    # Approximate decomposition (same algebra as homography case)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    K_inv = np.linalg.inv(K)
    B = K_inv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    lam = 0.5 * (1.0 / (np.linalg.norm(b1) + eps) + 1.0 / (np.linalg.norm(b2) + eps))

    r1 = lam * b1
    r2 = lam * b2
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack([r1, r2, r3])

    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = (lam * b3).reshape(3, 1)

    # in-front constraint
    if t[2, 0] < 0:
        R[:, 0] *= -1
        R[:, 1] *= -1
        t *= -1

    return QRtransform(R, t)
