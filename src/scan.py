import cv2
import numpy as np
from typing import Optional, Tuple
from marker import MarkerData
from mathutils import *
from camera import CamData

def scan_img(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera: CamData,
    len_marker: float,
):
    marker_1 = None
    marker_2 = None
    marker_3 = None

    # If calibration data is not available, should set manually.
    # K_mat = get_camera_intrinsic(fx, fy, cx, cy).astype(np.float64)
    # dist_coeff = np.zeros((4, 1), dtype=np.float64)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3D object points (Z=0 plane)
    obj_corner = np.array([
        [-len_marker / 2.0, -len_marker / 2.0, 0.0],
        [-len_marker / 2.0,  len_marker / 2.0, 0.0],
        [ len_marker / 2.0,  len_marker / 2.0, 0.0],
        [ len_marker / 2.0, -len_marker / 2.0, 0.0],
    ], dtype=np.float64)

    img_corners, ids, rejected = detector.detectMarkers(img_gray)

    if ids is None or len(ids) == 0:
        return marker_1, marker_2, marker_3

    for id_arr, img_corner in zip(ids, img_corners):
        marker_id = int(id_arr[0])  # ids shape (N,1)

        # img_corner is typically (1,4,2) -> reshape to (4,2)
        img_pts = img_corner.reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_corner,
            img_pts,
            camera.camera_mat,
            camera.distortion_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  # robust method for planar squares
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)  # R: (3,3)
        t = tvec.reshape(3, 1)      # (3,1)

        m = MarkerData(index=marker_id, corner_pos=img_pts, orientation=Orientation(R, t))

        if marker_id == 1:
            marker_1 = m
        elif marker_id == 2:
            marker_2 = m
        elif marker_id == 3:
            marker_3 = m

    return marker_1, marker_2, marker_3



if __name__ == "__main__":
    # img_calibrate = cv2.imread('images/calibrate_img.jpg')
    img_new = cv2.imread('images/arucoimg.png')
    

    # Camera intrinsic parameters
    len_marker = 0.1
    h, w = img_new.shape[:2]
    cx, cy = w/2, h/2
    fx = max(w, h)
    fy = max(w, h)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) 
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary = aruco_dict, detectorParams=parameters)

    # scan existing qr codes and print
    marker_1, marker_2, marker_3 = scan_img(img_new, detector, len_marker, fx, fy, cx, cy)
    print(marker_1)

    # # calibrate to set zero angle position
    # T_12_zero = inverse_transform(get_21_transform(marker_1.orientation, marker_2.orientation))

    # # read new state to get rotation
    # marker_1, marker_2 = scan_img(img_new, len_marker, fx, fy, cx, cy)
    # T_21_new = get_21_transform(marker_1.orientation, marker_2.orientation)
    # T_2new2 = multiple_transform(T_21_new, T_12_zero)

    # print(f"\n==========================\n{T_2new2.rot}")
    # print(f"\n==========================\n{T_2new2.trans}")
    # print(f"\n==========================\n{angle_from_transform(T_2new2)*180/3.14}")