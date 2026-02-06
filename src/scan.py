import cv2
import numpy as np
from typing import Optional, Tuple
from markerdata import MarkerData
from mathutils import *
from ArUco.utils import ARUCO_DICT, aruco_display

def scan_img(
    image: np.ndarray, 
    len_marker: float, 
    fx: float, fy: float, 
    cx: float, cy: float
)-> Tuple[Optional[Orientation], Optional[Orientation]]:
    
    marker_1 = None
    marker_2 = None
    marker_3 = None

    K_mat = get_camera_intrinsic(fx, fy, cx, cy)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters, cameraMatrix=K_mat)

    for index, corner in zip(ids, corners):
        if index == 1:
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, len_marker, K_mat)
            marker_1 = MarkerData(index=index, corner_pos=corner, orientation=Orientation(cv2.Rodrigues(rvec), tvec.T))

        elif index == 2:
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, len_marker, K_mat)
            marker_1 = MarkerData(index=index, corner_pos=corner, orientation=Orientation(cv2.Rodrigues(rvec), tvec.T))

        elif index == 3:
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corner, len_marker, K_mat)
            marker_1 = MarkerData(index=index, corner_pos=corner, orientation=Orientation(cv2.Rodrigues(rvec), tvec.T))

        # elif decoded == "position2":
        #     pass

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

    # scan existing qr codes and print
    marker_1, marker_2, marker_3 = scan_img(img_new, len_marker, fx, fy, cx, cy)
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