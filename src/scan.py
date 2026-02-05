from qreader import QReader
import cv2
import numpy as np
from qrtransforms import calibrate_perspective, get_orientation, get_camera_intrinsic, QRdata, inverse_transform, multiple_transform, QRtransform
# from qrdata import 


def scan_qr(image, detector, len_qr, fx, fy, cx, cy):
    decodedQRs, QRlocations = detector.detect_and_decode(image=image, return_detections=True)

    for (decodedQR, QRlocation) in zip(decodedQRs, QRlocations):
        if decodedQR == 'position1':
            qr_1 = QRdata(
                decoded_qr = decodedQR,
                center_pos = np.array(QRlocation['cxcy']),
                corner_pos = QRlocation['padded_quad_xy'],
                orientation = get_orientation(QRlocation['padded_quad_xy'], get_camera_intrinsic(fx, fy, cx, cy), len_qr) 
            )
            print(qr_1)

        elif decodedQR == 'position2':
            qr_2 = QRdata(
                decoded_qr = decodedQR,
                center_pos = np.array(QRlocation['cxcy']),
                corner_pos = QRlocation['padded_quad_xy'],
                orientation = get_orientation(QRlocation['padded_quad_xy'], get_camera_intrinsic(fx, fy, cx, cy), len_qr) 
            )
            print(qr_2)
        
        else:
            print(f"Unknown QR detected: {decodedQR}")
            print(f"position: x: {QRlocation['cxcy'][0]}, y: {QRlocation['cxcy'][1]}")
    
    return qr_1, qr_2


def get_21_transform(transform_c1, transform_c2):
    T_2c = inverse_transform(transform_c2)
    T_c1 = transform_c1

    T_21 = multiple_transform(T_2c, T_c1)

    return T_21
    

if __name__ == "__main__":
    img_calibrate = cv2.imread('images/plateimg.png')
    img_new = cv2.imread('images/plateimg2.png')
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
