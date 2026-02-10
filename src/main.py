# main.py
import sys
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
from mathutils import Orientation, get_camera_intrinsic, inverse_transform, multiple_transform, get_21_transform, angle_from_transform


def _as_col(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape == (3, 1):
        return v
    v = v.reshape(-1)
    if v.size == 3:
        return v.reshape(3, 1)
    return np.zeros((3, 1), dtype=np.float64)


def _se3_to_text(T: Orientation, name: str) -> str:
    R = np.asarray(T.rot, dtype=np.float64).reshape(3, 3)
    t = _as_col(T.trans)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3:4] = t
    return f"{name} =\n{np.array2string(M, precision=4, suppress_small=True)}"


def _project_points(Xc: np.ndarray, fx: float, fy: float, cx: float, cy: float, eps: float = 1e-9) -> np.ndarray:
    Xc = np.asarray(Xc, dtype=np.float64)
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < eps, eps, z)
    u = fx * (Xc[:, 0:1] / z) + cx
    v = fy * (Xc[:, 1:2] / z) + cy
    return np.hstack([u, v])


def _draw_quad(img_bgr: np.ndarray, quad_xy: np.ndarray, color: tuple[int, int, int], thickness: int = 2):
    q = np.asarray(quad_xy, dtype=np.float64).reshape(4, 2)
    pts = q.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i in range(4):
        p = tuple(np.round(q[i]).astype(int))
        cv2.circle(img_bgr, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)


def _draw_axes(img_bgr: np.ndarray, T_marker_cam: Orientation, fx: float, fy: float, cx: float, cy: float, axis_len: float):
    R = np.asarray(T_marker_cam.rot, dtype=np.float64).reshape(3, 3)
    t = _as_col(T_marker_cam.trans)

    O = np.array([[0.0, 0.0, 0.0]], dtype=np.float64).T
    X = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64).T
    Y = np.array([[0.0, axis_len, 0.0]], dtype=np.float64).T
    Z = np.array([[0.0, 0.0, axis_len]], dtype=np.float64).T

    Pw = np.hstack([O, X, Y, Z])      # marker frame points
    Pc = (R @ Pw) + t                 # camera frame points

    PcT = Pc.T
    if np.any(PcT[:, 2] <= 1e-6):
        return

    uv = _project_points(PcT, fx, fy, cx, cy).astype(np.int32)
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


def _track_quad(prev_gray: np.ndarray, gray: np.ndarray, quad_xy: np.ndarray):
    p0 = np.asarray(quad_xy, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if p1 is None or st is None:
        return None
    st = st.reshape(-1)
    if int(st.sum()) < 4:
        return None
    return p1.reshape(4, 2)


class DetectionThread(QThread):
    resultReady = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, frame_bgr: np.ndarray, detector: aruco.ArucoDetector, len_marker: float,
                 fx: float, fy: float, cx: float, cy: float):
        super().__init__()
        self.frame_bgr = frame_bgr
        self.detector = detector
        self.len_marker = float(len_marker)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def run(self):
        try:
            m1, m2, m3 = scan_img(
                image=self.frame_bgr,
                detector=self.detector,
                len_marker=self.len_marker,
                fx=self.fx, fy=self.fy,
                cx=self.cx, cy=self.cy,
            )
            self.resultReady.emit(m1, m2, m3)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
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

        self.len_marker = 0.03

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.detect_every = 4
        self.frame_idx = 0
        self.detect_thread = None

        self.prev_gray = None

        self.m1 = None
        self.m2 = None
        self.m3 = None

        self.quad1 = None
        self.quad2 = None
        self.quad3 = None

        self.pose1 = None
        self.pose2 = None
        self.pose3 = None

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
        self.timer.start(30)

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

    def _ensure_intrinsics(self, frame_bgr: np.ndarray):
        if self.fx is not None:
            return
        h, w = frame_bgr.shape[:2]
        self.cx = w * 0.5
        self.cy = h * 0.5
        f = float(max(w, h))
        self.fx = f
        self.fy = f

    def _start_detection(self, frame_bgr: np.ndarray):
        if self.detect_thread is not None and self.detect_thread.isRunning():
            return
        self.detect_thread = DetectionThread(
            frame_bgr=frame_bgr.copy(),
            detector=self.detector,
            len_marker=self.len_marker,
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
        )
        self.detect_thread.resultReady.connect(self._on_detection_result)
        self.detect_thread.error.connect(self._on_detection_error)
        self.detect_thread.start()

    def _on_detection_error(self, msg: str):
        pass

    def _on_detection_result(self, m1, m2, m3):
        self.m1, self.m2, self.m3 = m1, m2, m3

        if m1 is not None:
            self.quad1 = np.asarray(m1.corner_pos, dtype=np.float64).reshape(4, 2)
            self.pose1 = m1.orientation
        else:
            self.quad1 = None
            self.pose1 = None

        if m2 is not None:
            self.quad2 = np.asarray(m2.corner_pos, dtype=np.float64).reshape(4, 2)
            self.pose2 = m2.orientation
        else:
            self.quad2 = None
            self.pose2 = None

        if m3 is not None:
            self.quad3 = np.asarray(m3.corner_pos, dtype=np.float64).reshape(4, 2)
            self.pose3 = m3.orientation
        else:
            self.quad3 = None
            self.pose3 = None

    def on_calibrate(self):
        try:
            if self.pose1 is None or self.pose2 is None:
                self.T_12_zero = None
                self.lbl_rel.setText("relative (after calib):\nNeed both marker 1 and 2 visible.")
                return

            T_21 = get_21_transform(self.pose1, self.pose2)   # 1 -> 2(now)
            self.T_12_zero = inverse_transform(T_21)          # 2(now) -> 1
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

        self._ensure_intrinsics(frame)

        img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        do_detect = (self.frame_idx % self.detect_every == 0)
        if do_detect:
            self._start_detection(frame)
        else:
            if self.prev_gray is not None:
                if self.quad1 is not None:
                    q = _track_quad(self.prev_gray, gray, self.quad1)
                    if q is not None:
                        self.quad1 = q
                if self.quad2 is not None:
                    q = _track_quad(self.prev_gray, gray, self.quad2)
                    if q is not None:
                        self.quad2 = q
                if self.quad3 is not None:
                    q = _track_quad(self.prev_gray, gray, self.quad3)
                    if q is not None:
                        self.quad3 = q

        self.prev_gray = gray
        self.frame_idx += 1

        if self.quad1 is not None:
            _draw_quad(img, self.quad1, (255, 0, 255), 2)
        if self.pose1 is not None:
            _draw_axes(img, self.pose1, self.fx, self.fy, self.cx, self.cy, axis_len=self.len_marker * 0.5)

        if self.quad2 is not None:
            _draw_quad(img, self.quad2, (0, 165, 255), 2)
        if self.pose2 is not None:
            _draw_axes(img, self.pose2, self.fx, self.fy, self.cx, self.cy, axis_len=self.len_marker * 0.5)

        if self.quad3 is not None:
            _draw_quad(img, self.quad3, (255, 255, 0), 2)
        if self.pose3 is not None:
            _draw_axes(img, self.pose3, self.fx, self.fy, self.cx, self.cy, axis_len=self.len_marker * 0.5)

        if self.pose1 is not None:
            self.lbl_cam_p1.setText(_se3_to_text(self.pose1, "T_cam_p1"))
        else:
            self.lbl_cam_p1.setText("T_cam_p1 =\n-")

        if self.pose2 is not None:
            self.lbl_cam_p2.setText(_se3_to_text(self.pose2, "T_cam_p2"))
        else:
            self.lbl_cam_p2.setText("T_cam_p2 =\n-")

        if self.T_12_zero is not None and self.pose1 is not None and self.pose2 is not None:
            try:
                T_21_new = get_21_transform(self.pose1, self.pose2)
                T_2new2 = multiple_transform(T_21_new, self.T_12_zero)
                yaw_deg = float(angle_from_transform(T_2new2) * 180.0 / np.pi)
                t = _as_col(T_2new2.trans).reshape(3)
                self.lbl_rel.setText(
                    f"{_se3_to_text(T_2new2, 'T_2new2')}\n\n"
                    f"yaw(z) [deg]: {yaw_deg:.2f}\n"
                    f"translation [m]: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"
                )
            except Exception as e:
                self.lbl_rel.setText(f"relative (after calib):\nCompute error:\n{e}")
        else:
            if self.T_12_zero is None:
                self.lbl_rel.setText("relative (after calib):\nPress Calibration when marker 1 and 2 are visible.")
            else:
                self.lbl_rel.setText("relative (after calib):\nNeed both marker 1 and 2 visible.")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
