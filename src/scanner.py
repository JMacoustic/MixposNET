import cv2
import numpy as np
from typing import Optional, Tuple
from marker import MarkerData
from mathutils import *
from camera import CamData, load_camera
from pathlib import Path

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

        # img_corner (1,4,2) -> reshape to (4,2)
        img_pts = img_corner.reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_corner,
            img_pts,
            camera.camera_mat,
            camera.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec) 
        t = tvec.reshape(3, 1)

        m = MarkerData(index=marker_id, corner_pos=img_pts, orientation=Orientation(R, t))

        if marker_id == 1:
            marker_1 = m
        elif marker_id == 2:
            marker_2 = m
        elif marker_id == 3:
            marker_3 = m

    return marker_1, marker_2, marker_3

def set_zero_img(
    zero_image: np.ndarray,
    cam_name: str,
    aruco_dict: str,
    len_marker: float
):
    """
    Save zero position of the mixer from list of images. 

    Inputs
     - zero_image: BGR numpy image of the mixer zero position. All 3 markers should appear clearly
     - camera: name of the camera. should be consistent for zeroing and detecting rotation 
     - aruco_dict: key for the aruco marker dictionary
     - len_marker: side length of the aruco marker

    Returns
     - relative orientation between marker1 & 2
     - relative orientation between marker1 & 3
    """
    CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"    
    camera = load_camera(CAM_NPZ_PATH)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=parameters)

    marker_1, marker_2, marker_3 = scan_img(zero_image, detector, camera, len_marker)

    T_12zero = None
    T_13zero = None 

    if marker_1 is None or marker_2 is None or marker_3 is None:
        raise RuntimeError("Unable to set zero. Make sure all 3 markers are shown in the image")
    else:
        T_21 = get_21_transform(marker_1.orientation, marker_2.orientation)
        T_12zero = inverse_transform(T_21)
        T_31 = get_21_transform(marker_1.orientation, marker_3.orientation)
        T_13zero = inverse_transform(T_31)
    
    
    return T_12zero, T_13zero

        
def get_rotation_img(
    image: np.ndarray,
    zero_pos_12: Orientation,
    zero_pos_13: Orientation,
    cam_name: str,
    aruco_dict: str,
    len_marker: float
):
    """
    Calculate mixer rotation from the zero position and the input image 

    Inputs
     - image: BGR numpy image of the rotated mixer
     - zero_pos_12: relative transform between marker 1 & 2
     - zero_pos_13: relative transform between marker 1 & 3
     - cam_name: name of the camera. should be consistent for zeroing and detecting rotation 
     - aruco_dict: key for the aruco marker dictionary
     - len_marker: side length of the aruco marker

    Returns
     - mixer rotation angle in degree
    """

    CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"    
    camera = load_camera(CAM_NPZ_PATH)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=parameters)

    marker_1, marker_2, marker_3 = scan_img(image, detector, camera, len_marker)

    T_2new2zero = None
    T_3new3zero = None 
    T_12zero = zero_pos_12
    T_13zero = zero_pos_13

    if marker_1 is None:
        raise RuntimeError("Marker 1 not detected.")
    elif marker_2 is not None:
        T_2new1 = get_21_transform(marker_1.orientation, marker_2.orientation)
        T_2new2zero = multiple_transform(T_2new1, T_12zero)
    elif marker_3 is not None:
        T_3new1 = get_21_transform(marker_1.orientation, marker_3.orientation)
        T_3new3zero = multiple_transform(T_3new1, T_13zero)

    T_newzero, info = pick_or_fuse(T_3new3zero, T_2new2zero)
    angle = angle_from_transform(T_newzero)

    return float(np.degrees(angle))


if __name__ == "__main__":
    # img_calibrate = cv2.imread('images/calibrate_img.jpg')
    img_new = cv2.imread('images/test1.jpg')
    img_calibrate = cv2.imread('images/test2.jpg')
    
    len_marker = 16

    T_12zero, T_13zero = set_zero_img(img_calibrate, "frontcam", "DICT_4X4_100", len_marker)

    angle = get_rotation_img(img_new, T_12zero, T_13zero, "frontcam", "DICT_4X4_100", len_marker)

    print(angle)

    # CAM_NAME = "frontcam"
    # CAM_NPZ_PATH = Path("src") / "camdata" / CAM_NAME / f"{CAM_NAME}.npz"    
    # camera = load_camera(CAM_NPZ_PATH)

    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) 
    # parameters = cv2.aruco.DetectorParameters()
    # detector = cv2.aruco.ArucoDetector(dictionary = aruco_dict, detectorParams=parameters)

    # # scan existing qr codes and print
    # marker_1, marker_2, marker_3 = scan_img(img_calibrate, detector, camera, len_marker)
    # print(marker_1)
    # print(marker_2)

    # T_12_zero = inverse_transform(get_21_transform(marker_1.orientation, marker_2.orientation))

    # marker_1, marker_2, marker_3 = scan_img(img_new, detector, camera, len_marker)
    # print(marker_2)

    # T_21_new = get_21_transform(marker_1.orientation, marker_2.orientation)
    # T_2new2 = multiple_transform(T_21_new, T_12_zero)

    # print(f"\n==========================\n{T_2new2.rot}")
    # print(f"\n==========================\n{T_2new2.trans}")
    # print(f"\n==========================\n{angle_from_transform(T_2new2)*180/3.14}")

    