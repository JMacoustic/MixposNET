# main.py
import sys
from pathlib import Path
import numpy as np
import cv2
from cv2 import aruco

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QGroupBox,
)

from scan import scan_img
from camera import CamData
from marker import MarkerData
from mathutils import Orientation, inverse_transform, multiple_transform, angle_from_transform, get_21_transform


# =========================
# Camera loader (NPZ -> CamData)
# =========================
def load_camera_npz(npz_path: str | Path) -> CamData:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    name = str(data.get("name", npz_path.stem))
    index = int(data.get("index", 0))

    K = np.asarray(data["camera_mat"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    return CamData(index=index, name=name, camera_mat=K, distortion_coeffs=dist)


def _as_col(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape == (3,):
        return v.reshape(3, 1)
    if v.shape == (3, 1):
        return v
    v = v.reshape(-1)
    if v.size == 3:
        return v.reshape(3, 1)
    return np.zeros((3, 1), dtype=np.float64)


def _se3_to_text(T: Orientation, name: str = "T") -> str:
    R = np.asarray(T.rot, dtype=np.float64).reshape(3, 3)
    t = _as_col(T.trans)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3:4] = t
    return f"{name} =\n{np.array2string(M, precision=4, suppress_small=True)}"


def _project_points(Xc: np.ndarray, K: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    Xc = np.asarray(Xc, dtype=np.float64)
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < eps, eps, z)
    u = K[0, 0] * (Xc[:, 0:1] / z) + K[0, 2]
    v = K[1, 1] * (Xc[:, 1:2] / z) + K[1, 2]
    return np.hstack([u, v])


def _draw_quad(img_bgr: np.ndarray, quad_xy: np.ndarray, color: tuple[int, int, int], thickness: int = 2):
    q = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
    pts = q.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i in range(4):
        p = tuple(np.round(q[i]).astype(int))
        cv2.circle(img_bgr, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)


def _draw_axes(img_bgr: np.ndarray, T_obj_cam: Orientation, K: np.ndarray, axis_len: float):
    R = np.asarray(T_obj_cam.rot, dtype=np.float64).reshape(3, 3)
    t = _as_col(T_obj_cam.trans)

    O = np.array([[0.0, 0.0, 0.0]], dtype=np.float64).T
    X = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64).T
    Y = np.array([[0.0, axis_len, 0.0]], dtype=np.float64).T
    Z = np.array([[0.0, 0.0, axis_len]], dtype=np.float64).T

    Pw = np.hstack([O, X, Y, Z])
    Pc = (R @ Pw) + t

    PcT = Pc.T
    if np.any(PcT[:, 2] <= 1e-6):
        return

    uv = _project_points(PcT, K).astype(np.int32)
    o = tuple(uv[0])
    px = tuple(uv[1])
    py = tuple(uv[2])
    pz = tuple(uv[3])

    cv2.line(img_bgr, o, px, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, py, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(img_bgr, o, pz, (255, 0, 0), 2, lineType=cv2.LINE_AA)

    cv2.putText(img_bgr, "x'", px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "y'", py, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "z'", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


class DetectionThread(QThread):
    resultReady = Signal(object, object, object)  # (m1, m2, m3)
    error = Signal(str)

    def __init__(self, frame_bgr: np.ndarray, detector: aruco.ArucoDetector, camera: CamData, len_marker: float):
        super().__init__()
        self.frame_bgr = frame_bgr
        self.detector = detector
        self.camera = camera
        self.len_marker = float(len_marker)

    def run(self):
        try:
            m1, m2, m3 = scan_img(
                image=self.frame_bgr,
                detector=self.detector,
                camera=self.camera,
                len_marker=self.len_marker,
            )
            self.resultReady.emit(m1, m2, m3)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, camera: CamData):
        super().__init__()
        self.setWindowTitle("ArUco SE(3) Tracker")

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=params)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera = camera
        self.K = np.asarray(self.camera.camera_mat, dtype=np.float64).reshape(3, 3)

        self.len_marker = 0.03
        self.T_12_zero = None

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        self.lbl_cam_p1 = QLabel("cam-position1:\n-")
        self.lbl_cam_p2 = QLabel("cam-position2:\n-")
        self.lbl_rel = QLabel("relative (after calib):\n-")
        for w in (self.lbl_cam_p1, self.lbl_cam_p2, self.lbl_rel):
            w.setFont(mono)
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            w.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.btn_calib = QPushButton("Calibration (Zero position2)")
        self.btn_calib.clicked.connect(self.on_calibrate)

        side = QWidget()
        side_layout = QVBoxLayout(side)

        grp1 = QGroupBox("SE(3) Matrices")
        grp1_l = QVBoxLayout(grp1)
        grp1_l.addWidget(self.lbl_cam_p1)
        grp1_l.addWidget(self.lbl_cam_p2)

        grp2 = QGroupBox("Relative movement (position2)")
        grp2_l = QVBoxLayout(grp2)
        grp2_l.addWidget(self.btn_calib)
        grp2_l.addWidget(self.lbl_rel)

        side_layout.addWidget(grp1)
        side_layout.addWidget(grp2)
        side_layout.addStretch(1)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.video_label, stretch=4)
        layout.addWidget(side, stretch=1)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        self.detect_thread = None

        self.m1_last = None
        self.m2_last = None
        self.m3_last = None

        self.quad1 = None
        self.quad2 = None

        self.pose1 = None
        self.pose2 = None

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.detect_thread is not None and self.detect_thread.isRunning():
                self.detect_thread.quit()
                self.detect_thread.wait(200)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)

    def _start_detection(self, frame_bgr: np.ndarray):
        if self.detect_thread is not None and self.detect_thread.isRunning():
            return

        self.detect_thread = DetectionThread(
            frame_bgr=frame_bgr.copy(),
            detector=self.detector,
            camera=self.camera,
            len_marker=self.len_marker,
        )
        self.detect_thread.resultReady.connect(self._on_detection_result)
        self.detect_thread.error.connect(self._on_detection_error)
        self.detect_thread.start()

    def _on_detection_error(self, msg: str):
        pass

    def _on_detection_result(self, m1, m2, m3):
        self.m1_last, self.m2_last, self.m3_last = m1, m2, m3

        if m1 is not None:
            self.quad1 = np.asarray(m1.corner_pos, dtype=np.float64).reshape(4, 2)
            self.pose1 = m1.orientation

        if m2 is not None:
            self.quad2 = np.asarray(m2.corner_pos, dtype=np.float64).reshape(4, 2)
            self.pose2 = m2.orientation

    def on_calibrate(self):
        try:
            if self.pose1 is None or self.pose2 is None:
                self.T_12_zero = None
                self.lbl_rel.setText("relative (after calib):\nNeed both marker 1 and marker 2 visible.")
                return

            T_21 = get_21_transform(self.pose1, self.pose2)  # 1 -> 2(now)
            self.T_12_zero = inverse_transform(T_21)         # 2(now) -> 1
            self.lbl_rel.setText("relative (after calib):\nCalibrated. Move marker 2 now.")
        except Exception as e:
            self.T_12_zero = None
            self.lbl_rel.setText(f"relative (after calib):\nCalibration error:\n{e}")

    def update_frame(self):
        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return
        except Exception:
            return

        img = frame.copy()

        self._start_detection(frame)

        if self.quad1 is not None:
            _draw_quad(img, self.quad1, (255, 0, 255), 2)
        if self.pose1 is not None:
            _draw_axes(img, self.pose1, self.K, axis_len=self.len_marker * 0.5)

        if self.quad2 is not None:
            _draw_quad(img, self.quad2, (0, 165, 255), 2)
        if self.pose2 is not None:
            _draw_axes(img, self.pose2, self.K, axis_len=self.len_marker * 0.5)

        if self.pose1 is not None:
            self.lbl_cam_p1.setText(_se3_to_text(self.pose1, "T_cam_m1"))
        else:
            self.lbl_cam_p1.setText("T_cam_m1 =\n-")

        if self.pose2 is not None:
            self.lbl_cam_p2.setText(_se3_to_text(self.pose2, "T_cam_m2"))
        else:
            self.lbl_cam_p2.setText("T_cam_m2 =\n-")

        if self.T_12_zero is not None and self.pose1 is not None and self.pose2 is not None:
            try:
                T_21_new = get_21_transform(self.pose1, self.pose2)    # 1 -> 2(new)
                T_2new2 = multiple_transform(T_21_new, self.T_12_zero) # 2(old)->2(new)
                yaw_deg = float(angle_from_transform(T_2new2) * 180.0 / np.pi)
                t = _as_col(T_2new2.trans).reshape(3)
                self.lbl_rel.setText(
                    "T_2new2:\n"
                    f"{_se3_to_text(T_2new2, 'T_2new2')}\n\n"
                    f"yaw(z) [deg]: {yaw_deg:.2f}\n"
                    f"translation: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"
                )
            except Exception as e:
                self.lbl_rel.setText(f"relative (after calib):\nCompute error:\n{e}")
        else:
            if self.T_12_zero is None:
                self.lbl_rel.setText("relative (after calib):\nPress Calibration when both markers are visible.")
            else:
                self.lbl_rel.setText("relative (after calib):\nNeed both marker 1 and marker 2 visible.")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass


def main():
    # ---- configure these ----
    CAM_NAME = "frontcam"  # folder: src/camdata/<CAM_NAME>/ and file: <CAM_NAME>.npz
    CAM_NPZ_PATH = Path("src") / "camdata" / CAM_NAME / f"{CAM_NAME}.npz"
    # ------------------------

    try:
        camera = load_camera_npz(CAM_NPZ_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load camera npz: {CAM_NPZ_PATH}\n{e}")

    app = QApplication(sys.argv)
    win = MainWindow(camera=camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
