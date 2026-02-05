from model.qreader_moon import QReader
import cv2
import numpy as np
from qrtransforms import calibrate_perspective, get_orientation, get_camera_intrinsic, QRdata, inverse_transform, multiple_transform, QRtransform
from gpt_transform import *


def scan_qr(image, detector, len_qr, fx, fy, cx, cy):
    qr_1 = None
    qr_2 = None

    K = get_camera_intrinsic(fx, fy, cx, cy)

    # expects your modified QReader.detect_and_decode() that returns QRZbarInfo objects
    infos, detections = detector.detect_and_decode(image=image, return_detections=True)

    for info, det in zip(infos, detections):
        decoded = info.decoded
        zori = info.zbar_orientation

        if decoded == "position1":
            quad = det["padded_quad_xy"]
            orientation = square_pose_from_4pts(quad, K, len_qr)

            qr_1 = QRdata(
                decoded_qr=decoded,
                center_pos=np.asarray(det["cxcy"], dtype=np.float64),
                corner_pos=np.asarray(quad, dtype=np.float64),
                orientation=orientation,
            )
            print(qr_1)

        elif decoded == "position2":
            quad_raw = det["padded_quad_xy"]
            quad = reorder_quad_by_zbar_orientation(quad_raw, zori)
            orientation = square_pose_from_4pts(quad, K, len_qr)

            qr_2 = QRdata(
                decoded_qr=decoded,
                center_pos=np.asarray(det["cxcy"], dtype=np.float64),
                corner_pos=np.asarray(quad, dtype=np.float64),
                orientation=orientation,
            )
            print(qr_2)

        elif decoded is None:
            print("QR detected but not enough confidence")
            pass

        else:
            print(f"Unknown QR detected: {decoded}")
            print(f"position: x: {det['cxcy'][0]}, y: {det['cxcy'][1]}")
            print(f"zbar_orientation: {zori}")

    return qr_1, qr_2


def get_21_transform(transform_c1, transform_c2):
    T_2c = inverse_transform(transform_c2)
    T_c1 = transform_c1

    T_21 = multiple_transform(T_2c, T_c1)

    return T_21
    

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
    print(f"\n==========================\n{transform_to_angle(T_2new2)}")
