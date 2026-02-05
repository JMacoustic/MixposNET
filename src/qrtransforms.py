# Referenced from QReader src code. https://github.com/Eric-Canas/QReader
import numpy as np
import cv2
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass

class QRtransform:
    def __init__(self, rotation: np.ndarray = np.identity(3), translation: np.ndarray = np.zeros((3, 1))):
        self.rot = rotation
        self.trans = translation
    
    def reset(self):
        self.rot = np.identity(3)
        self.trans = np.zeros((3, 1))
  

@dataclass(frozen=True)
class QRdata:
    decoded_qr: str #= "Empty"
    center_pos: np.ndarray #= np.zeros(2)
    corner_pos: np.ndarray #= np.zeros((4, 2))
    orientation: QRtransform #= QRtransform(np.identity(3), np.zeros((3, 1)))

    def __str__(self) -> str:
        def fmt(arr: np.ndarray) -> str:
            return np.array2string(
                arr,
                precision=4,
                suppress_small=True,
                separator=", "
            )

        return (
            "\n"
            f"========== {self.decoded_qr} ==========\n"
            f"  center : {fmt(self.center_pos)},\n"
            f"  corners :\n{fmt(self.corner_pos)},\n"
            f"  rotation :\n{fmt(self.orientation.rot)},\n"
            f"  translation :\n{fmt(self.orientation.trans)}\n"
        )


def inverse_transform(T: QRtransform):
    R_mat = T.rot
    P_vec = T.trans

    R_inv = R_mat.T
    P_inv = - np.matmul(R_inv, P_vec)

    return QRtransform(R_inv, P_inv)


def multiple_transform(T1: QRtransform, T2: QRtransform):
    R_mat1 = T1.rot
    P_vec1 = T1.trans
    R_mat2 = T2.rot
    P_vec2 = T2.trans

    R_mul = np.matmul(R_mat1, R_mat2)
    P_mul = np.matmul(R_mat1, P_vec2) + P_vec1

    return QRtransform(R_mul, P_mul)


def calibrate_perspective(
    image: np.ndarray, padded_quad_xy: np.ndarray
) -> np.ndarray:
    
    N = _calculate_max_size(padded_quad_xy)
    H_mat = get_homography(padded_quad_xy)

    dst_img = cv2.warpPerspective(image, H_mat, (N, N))

    return dst_img


def get_homography(
    padded_quad_xy: np.ndarray
) -> np.ndarray:
    # Create destination points for the perspective transform. This forms an N x N square
    N = _calculate_max_size(padded_quad_xy)

    dst_pts = np.array(
        [[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]], dtype=np.float32
    )

    # Compute the perspective transform matrix
    H_mat = cv2.getPerspectiveTransform(padded_quad_xy, dst_pts)

    return H_mat


def get_orientation(
    padded_quad_xy: np.ndarray,
    camera_intrinsic: np.ndarray,
    len_qr: float
) -> QRtransform:
    # img_quad_xy: (4,2) corners in image pixels, consistent order with obj_pts
    obj_pts = np.array([
        [-len_qr/2,   -len_qr/2, 0],
        [-len_qr/2,   len_qr/2, 0],
        [len_qr/2,   len_qr/2, 0],
        [len_qr/2,   -len_qr/2, 0],
    ], dtype=np.float32)

    img_pts = padded_quad_xy.astype(np.float32)

    dist = np.zeros((4,1), dtype=np.float32)  # if you don't have distortion

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_intrinsic, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE  # excellent for planar squares
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    return QRtransform(R.astype(np.float64), tvec.astype(np.float64))


 


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


def _calculate_max_size(
    padded_quad_xy: np.ndarray
) -> int:
    
    # Define the width and height of the quadrilateral
    width1 = np.sqrt(
        ((padded_quad_xy[0][0] - padded_quad_xy[1][0]) ** 2)
        + ((padded_quad_xy[0][1] - padded_quad_xy[1][1]) ** 2)
    )
    width2 = np.sqrt(
        ((padded_quad_xy[2][0] - padded_quad_xy[3][0]) ** 2)
        + ((padded_quad_xy[2][1] - padded_quad_xy[3][1]) ** 2)
    )

    height1 = np.sqrt(
        ((padded_quad_xy[0][0] - padded_quad_xy[3][0]) ** 2)
        + ((padded_quad_xy[0][1] - padded_quad_xy[3][1]) ** 2)
    )
    height2 = np.sqrt(
        ((padded_quad_xy[1][0] - padded_quad_xy[2][0]) ** 2)
        + ((padded_quad_xy[1][1] - padded_quad_xy[2][1]) ** 2)
    )

    # Take the maximum width and height to ensure no information is lost
    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))
    N = max(max_width, max_height)

    return N


def get_orientation_back(
    padded_quad_xy: np.ndarray,
    camera_intrinsic: np.ndarray,
) -> QRtransform:
    """
    :param padded_quad_xy: (4, 2) float32 array of plane points (X, Y) with Z=0
    :param camera_intrinsic: (3, 3) camera intrinsic matrix K
    :return: (3, 3) rotation matrix and (3) translation matrix from qr plane to camera frame
    """
    K_mat = camera_intrinsic
    H_mat = get_homography(padded_quad_xy)

    _, Rs, Ts, _ = cv2.decomposeHomographyMat(H_mat, K_mat)

    # plane_pts = np.hstack([padded_quad_xy, np.zeros((padded_quad_xy.shape[0], 1), dtype=np.float32), ])
    plane_pts = np.array([
        [865.6,   865.6, 0],
        [865.6,   1365.6, 0],
        [1365.6,   1365.6, 0],
        [1365.6,   865.6, 0],
    ], dtype=np.float32)

    best_idx = None
    best_positive_count = -1

    for i, (R, t) in enumerate(zip(Rs, Ts)):
        X_cam = (R @ plane_pts.T + t).T

        positive_depth_count = np.sum(X_cam[:, 2] > 0)

        if positive_depth_count > best_positive_count:
            best_positive_count = positive_depth_count
            best_idx = i

    if best_idx is None or best_positive_count == 0:
        raise RuntimeError("No valid homography decomposition found (all points behind camera).")

    return QRtransform(Rs[best_idx], Ts[best_idx])