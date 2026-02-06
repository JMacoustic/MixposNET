from QReader.qreader_moon import QReader
import cv2
import numpy as np
from qrdata import *
from homography import *
from mathutils import *
from typing import Optional, Tuple

def scan_qr(
    image: np.ndarray, 
    detector: QReader, 
    len_qr: float, 
    fx: float, fy: float, 
    cx: float, cy: float
)-> Tuple[Optional[QRdata], Optional[QRdata]]:
    
    qr_1 = None
    qr_2 = None

    K = get_camera_intrinsic(fx, fy, cx, cy)

    infos, detections = detector.detect_and_decode(image=image, return_detections=True)

    for info, det in zip(infos, detections):
        decoded = info.decoded
        zori = info.zbar_orientation

        if decoded == "position1":
            quad_raw = det["quad_xy"]
            quad = reorder_quad(quad_raw, zori)
            orientation = pos_from_quad(quad, K, len_qr)

            qr_1 = QRdata(
                decoded_qr=decoded,
                center_pos=np.asarray(det["cxcy"], dtype=np.float64),
                corner_pos=np.asarray(quad, dtype=np.float64),
                orientation=orientation,
            )

        elif decoded == "position2":
            quad_raw = det["quad_xy"]
            quad = reorder_quad(quad_raw, zori)
            orientation = pos_from_quad(quad, K, len_qr)

            qr_2 = QRdata(
                decoded_qr=decoded,
                center_pos=np.asarray(det["cxcy"], dtype=np.float64),
                corner_pos=np.asarray(quad, dtype=np.float64),
                orientation=orientation,
            )

    return qr_1, qr_2


def get_21_transform(transform_c1: QRtransform, transform_c2: QRtransform) -> QRtransform:
    """Input 2 SE3 transforms that shares same reference frame. Returns relative SE3 transform between them"""
    T_2c = inverse_transform(transform_c2)
    T_c1 = transform_c1

    transform_21 = multiple_transform(T_2c, T_c1)

    return transform_21
    

if __name__ == "__main__":
    img_calibrate = cv2.imread('images/cleanimg1.png')
    img_new = cv2.imread('images/cleanimg2.png')
    detector = QReader()

    # Camera intrinsic parameters
    len_qr = 0.1
    h, w = img_calibrate.shape[:2]
    cx, cy = w/2, h/2
    fx = max(w, h)
    fy = max(w, h)

    # scan existing qr codes and print
    qr_1, qr_2 = scan_qr(img_calibrate, detector, len_qr, fx, fy, cx, cy)

    # calibrate to set zero angle position
    T_12_zero = inverse_transform(get_21_transform(qr_1.orientation, qr_2.orientation))

    # read new state to get rotation
    qr_1, qr_2 = scan_qr(img_new, detector, len_qr, fx, fy, cx, cy)
    T_21_new = get_21_transform(qr_1.orientation, qr_2.orientation)
    T_2new2 = multiple_transform(T_21_new, T_12_zero)

    print(f"\n==========================\n{T_2new2.rot}")
    print(f"\n==========================\n{T_2new2.trans}")
    print(f"\n==========================\n{angle_from_transform(T_2new2)*180/3.14}")