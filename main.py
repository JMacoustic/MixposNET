# main.py
import sys
import time
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer
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

from src.QReader.qreader_moon import QReader
from src.scan import scan_qr, get_21_transform
from src.qrdata import QRtransform, inverse_transform, multiple_transform
from src.mathutils import angle_from_transform


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


def _se3_to_text(T: QRtransform, name: str = "T") -> str:
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

    # draw colored corners to show ordering
    corner_colors = [
        (0, 0, 255),    # p0 red
        (0, 255, 0),    # p1 green
        (255, 0, 0),    # p2 blue
        (0, 255, 255),  # p3 yellow
    ]
    for i in range(4):
        p = tuple(np.round(q[i]).astype(int))
        cv2.circle(img_bgr, p, 5, corner_colors[i], -1, lineType=cv2.LINE_AA)


def _draw_axes(img_bgr: np.ndarray, T_qr_cam: QRtransform, fx: float, fy: float, cx: float, cy: float, axis_len: float):
    R = np.asarray(T_qr_cam.rot, dtype=np.float64).reshape(3, 3)
    t = _as_col(T_qr_cam.trans)

    # QR frame points (meters)
    O = np.array([[0.0, 0.0, 0.0]], dtype=np.float64).T
    X = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64).T
    Y = np.array([[0.0, axis_len, 0.0]], dtype=np.float64).T
    Z = np.array([[0.0, 0.0, axis_len]], dtype=np.float64).T

    Pw = np.hstack([O, X, Y, Z])  # (3,4)
    Pc = (R @ Pw) + t             # (3,4)

    PcT = Pc.T  # (4,3)
    if np.any(PcT[:, 2] <= 1e-6):
        return

    uv = _project_points(PcT, fx, fy, cx, cy).astype(np.int32)
    o = tuple(uv[0])
    px = tuple(uv[1])
    py = tuple(uv[2])
    pz = tuple(uv[3])

    cv2.line(img_bgr, o, px, (0, 0, 255), 2, lineType=cv2.LINE_AA)   # X red
    cv2.line(img_bgr, o, py, (0, 255, 0), 2, lineType=cv2.LINE_AA)   # Y green
    cv2.line(img_bgr, o, pz, (255, 0, 0), 2, lineType=cv2.LINE_AA)   # Z blue

    cv2.putText(img_bgr, "x'", px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "y'", py, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, "z'", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR SE(3) Tracker")

        self.detector = QReader()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.last_frame_time = time.time()

        self.len_qr = 0.10  # meters
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.T_12_zero = None  # QR2(old) -> QR1 (at calibration)

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

        self.qr1_last = None
        self.qr2_last = None

    def closeEvent(self, event):
        try:
            self.timer.stop()
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
        self.fx = float(max(w, h))
        self.fy = float(max(w, h))

    def on_calibrate(self):
        try:
            if self.qr1_last is None or self.qr2_last is None:
                self.T_12_zero = None
                self.lbl_rel.setText("relative (after calib):\nNeed both position1 and position2 visible.")
                return

            T_21 = get_21_transform(self.qr1_last.orientation, self.qr2_last.orientation)  # 1 -> 2(now)
            self.T_12_zero = inverse_transform(T_21)  # 2(now) -> 1
            self.lbl_rel.setText("relative (after calib):\nCalibrated. Move position2 now.")
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
        qr1 = None
        qr2 = None

        try:
            qr1, qr2 = scan_qr(
                image=img,
                detector=self.detector,
                len_qr=self.len_qr,
                fx=self.fx, fy=self.fy,
                cx=self.cx, cy=self.cy,
            )
        except Exception:
            qr1, qr2 = None, None

        self.qr1_last = qr1
        self.qr2_last = qr2

        # overlay drawings
        if qr1 is not None:
            _draw_quad(img, qr1.corner_pos, (255, 0, 255), 2)
            _draw_axes(img, qr1.orientation, self.fx, self.fy, self.cx, self.cy, axis_len=self.len_qr * 0.5)

        if qr2 is not None:
            _draw_quad(img, qr2.corner_pos, (0, 165, 255), 2)
            _draw_axes(img, qr2.orientation, self.fx, self.fy, self.cx, self.cy, axis_len=self.len_qr * 0.5)

        # side panel matrices
        if qr1 is not None:
            self.lbl_cam_p1.setText(_se3_to_text(qr1.orientation, "T_cam_p1"))
        else:
            self.lbl_cam_p1.setText("T_cam_p1 =\n-")

        if qr2 is not None:
            self.lbl_cam_p2.setText(_se3_to_text(qr2.orientation, "T_cam_p2"))
        else:
            self.lbl_cam_p2.setText("T_cam_p2 =\n-")

        # relative motion after calibration
        if self.T_12_zero is not None and qr1 is not None and qr2 is not None:
            try:
                T_21_new = get_21_transform(qr1.orientation, qr2.orientation)      # 1 -> 2(new)
                T_2new2 = multiple_transform(T_21_new, self.T_12_zero)             # 2(old)->2(new)
                yaw_deg = float(angle_from_transform(T_2new2) * 180.0 / np.pi)
                t = _as_col(T_2new2.trans).reshape(3)
                self.lbl_rel.setText(
                    "T_2new2:\n"
                    f"{_se3_to_text(T_2new2, 'T_2new2')}\n\n"
                    f"yaw(z) [deg]: {yaw_deg:.2f}\n"
                    f"translation [m]: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"
                )
            except Exception as e:
                self.lbl_rel.setText(f"relative (after calib):\nCompute error:\n{e}")
        else:
            if self.T_12_zero is None:
                self.lbl_rel.setText("relative (after calib):\nPress Calibration when both QRs are visible.")
            else:
                self.lbl_rel.setText("relative (after calib):\nNeed both position1 and position2 visible.")

        # show image
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
