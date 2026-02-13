import cv2
import numpy as np
from marker import MarkerData
from mathutils import *
from camera import CamData, load_camera
from pathlib import Path


class MixScanner:
    def __init__(
        self,
        cam_name: str = "frontcam",
        len_marker: float = 16,
        aruco_dict: str = "DICT_4X4_100",
    ):
        CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"    
        self.camera = load_camera(CAM_NPZ_PATH)

        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=parameters)

        self.len_marker = len_marker

        self.marker_1 = None
        self.marker_2 = None
        self.marker_3 = None

        self.obj_coners = np.array([
            [-len_marker / 2.0, -len_marker / 2.0, 0.0],
            [-len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0,  len_marker / 2.0, 0.0],
            [ len_marker / 2.0, -len_marker / 2.0, 0.0],
        ], dtype=np.float64)

        self.T_12zero = None 
        self.T_13zero = None

        self.T_new_zero = None
        self.angle = 0.0
    

    def set_camera(self, cam_name: str):
        """set new camera of the scanner from name string"""
        CAM_NPZ_PATH = Path("src") / "camdata" / cam_name / f"{cam_name}.npz"    
        self.camera = load_camera(CAM_NPZ_PATH)

    
    def scan_img(self, image: np.ndarray):
        """
        Read aruco code from the given image and save as markerdata

        Inputs
        - image: BGR numpy image to scan for markers.

        Returns
        - marker_1 data
        - marker_2 data
        - marker_3 data
        """

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_corners, ids, rejected = self.detector.detectMarkers(img_gray)

        self.marker_1 = None
        self.marker_2 = None
        self.marker_3 = None

        if ids is None or len(ids) == 0:
            return self.marker_1, self.marker_2, self.marker_3

        for id_arr, img_corner in zip(ids, img_corners):
            marker_id = int(id_arr[0])  # ids shape (N,1)

            # img_corner (1,4,2) -> reshape to (4,2)
            img_pts = img_corner.reshape(-1, 2).astype(np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                self.obj_coners,
                img_pts,
                self.camera.camera_mat,
                self.camera.distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            R, _ = cv2.Rodrigues(rvec) 
            t = tvec.reshape(3, 1)

            m = MarkerData(index=marker_id, corner_pos=img_pts, orientation=Orientation(R, t))

            if marker_id == 1:
                self.marker_1 = m
            elif marker_id == 2:
                self.marker_2 = m
            elif marker_id == 3:
                self.marker_3 = m

        return self.marker_1, self.marker_2, self.marker_3
    

    def set_zero(self, zero_image: np.ndarray):
        """
        Save zero position of the mixer from the image. 

        Inputs
        - zero_image: BGR numpy image of the mixer zero position. All 3 markers should appear clearly

        Returns
        - relative orientation between marker1 & 2
        - relative orientation between marker1 & 3
        """

        marker_1, marker_2, marker_3 = self.scan_img(zero_image)
        
        if marker_1 is None or marker_2 is None or marker_3 is None:
            raise RuntimeError("Unable to set zero. Make sure all 3 markers are shown in the image")
        else:
            T_21 = get_21_transform(marker_1.orientation, marker_2.orientation)
            self.T_12zero = inverse_transform(T_21)
            T_31 = get_21_transform(marker_1.orientation, marker_3.orientation)
            self.T_13zero = inverse_transform(T_31)

            self.set_current_transform(self)
        
        return self.T_12zero, self.T_13zero
    
    def set_current_transform(self):
        """calculate current mixer status and save"""
        T_2new2zero = None
        T_3new3zero = None

        if self.marker_1 is None:
            raise RuntimeError("Marker 1 not detected.")
        elif self.marker_2 is not None:
            T_2new1 = get_21_transform(self.marker_1.orientation, self.marker_2.orientation)
            T_2new2zero = multiple_transform(T_2new1, self.T_12zero)
        elif self.marker_3 is not None:
            T_3new1 = get_21_transform(self.marker_1.orientation, self.marker_3.orientation)
            T_3new3zero = multiple_transform(T_3new1, self.T_13zero)

        self.T_new_zero, info = pick_or_fuse(T_3new3zero, T_2new2zero)
        self.angle = angle_from_transform(self.T_new_zero)
    

    def get_rotation(self, image: np.ndarray):
        """
        Calculate mixer rotation from the zero position and the input image 

        Inputs
        - image: BGR numpy image of the rotated mixer

        Returns
        - mixer rotation angle in degree
        """
        self.scan_img(image)
        self.set_current_transform()

        return float(np.degrees(angle))


    def get_transform(self, image: np.ndarray):
        """
        Calculate mixer rotation from the zero position and the input image 

        Inputs
        - image: BGR numpy image of the rotated mixer

        Returns
        - transform status of the mixer relative to zero position
        """
        self.scan_img(image)
        self.set_current_transform()

        return self.T_new_zero
    

# do not use this function. Only used for gui.
def scan_img_gui(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera: CamData,
    len_marker: float,
):
    """All-in-one scanner function for GUI development"""
    
    marker_1 = None
    marker_2 = None
    marker_3 = None

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


if __name__ == "__main__":
    img_new = cv2.imread('images/test1.jpg')
    img_zero = cv2.imread('images/test2.jpg')
    
    len_marker = 16

    scanner = MixScanner(cam_name="frontcam", aruco_dict="DICT_4X4_100", len_marker=len_marker)

    scanner.set_zero(img_zero)
    angle = scanner.get_rotation(img_new)

    print(angle)


    