# main.py
import sys
from pathlib import Path

import numpy as np
import cv2
from cv2 import aruco

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QPlainTextEdit, 
    QSizePolicy,
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QMainWindow,
    QGroupBox,
)

from typing import Optional

from scanner import scan_img
from camera import CamData, load_camera
from mathutils import *


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
    t = as_col(T_obj_cam.trans)

    O = np.array([[0.0, 0.0, 0.0]], dtype=np.float64).T
    X = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64).T
    Y = np.array([[0.0, axis_len, 0.0]], dtype=np.float64).T
    Z = np.array([[0.0, 0.0, axis_len]], dtype=np.float64).T

    Pw = np.hstack([O, X, Y, Z])
    Pc = (R @ Pw) + t

    PcT = Pc.T
    if np.any(PcT[:, 2] <= 1e-6):
        return

    uv = project_points(PcT, K).astype(np.int32)
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


def _setup_console_box(w, mono: QFont, cols: int, lines: int):
    w.setFont(mono)
    w.setReadOnly(True)
    w.setLineWrapMode(QPlainTextEdit.NoWrap)
    w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    w.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    fm = QFontMetrics(mono)
    char_w = fm.horizontalAdvance("M")
    line_h = fm.lineSpacing()

    # small padding so text doesn't touch edges
    width = char_w * cols + 16
    height = line_h * lines + 16

    w.setFixedSize(width, height)


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

        self.cap = cv2.VideoCapture(camera.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera.index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera = camera
        self.K = np.asarray(self.camera.camera_mat, dtype=np.float64).reshape(3, 3)

        self.len_marker = 18

        self.T_12zero = None
        self.T_13zero = None
        self.T_plate_last = None

        self.rot_diff_thresh_deg = 20.0

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 480)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        self.txt_cam_p1 = QPlainTextEdit()
        self.txt_cam_p2 = QPlainTextEdit()
        self.txt_cam_p3 = QPlainTextEdit()
        self.txt_rel    = QPlainTextEdit()

        _setup_console_box(self.txt_cam_p1, mono, cols=44, lines=8)
        _setup_console_box(self.txt_cam_p2, mono, cols=44, lines=8)
        _setup_console_box(self.txt_cam_p3, mono, cols=44, lines=8)
        _setup_console_box(self.txt_rel,    mono, cols=44, lines=18)

        self.txt_cam_p1.setPlainText("cam-position1:\n-")
        self.txt_cam_p2.setPlainText("cam-position2:\n-")
        self.txt_cam_p3.setPlainText("cam-position3:\n-")
        self.txt_rel.setPlainText("relative (after zeroing):\n-")


        self.btn_zero = QPushButton("Zero plate position")
        self.btn_zero.clicked.connect(self.on_zeropos)

        side = QWidget()
        side_layout = QVBoxLayout(side)

        grp1 = QGroupBox("SE(3) Matrices")
        grp1_l = QVBoxLayout(grp1)
        grp1_l.addWidget(self.txt_cam_p1)
        grp1_l.addWidget(self.txt_cam_p2)
        grp1_l.addWidget(self.txt_cam_p3)

        grp2 = QGroupBox("Relative movement (plate)")
        grp2_l = QVBoxLayout(grp2)
        grp2_l.addWidget(self.btn_zero)
        grp2_l.addWidget(self.txt_rel)

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
        self.quad3 = None

        self.pose1 = None
        self.pose2 = None
        self.pose3 = None

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


    def on_zeropos(self):
        try:
            if self.pose1 is None or self.pose2 is None or self.pose3 is None:
                self.T_12zero = None
                self.T_13zero = None
                self.T_plate_last = None
                self.txt_rel.setPlainText("relative (after zero):\nNeed markers 1, 2, 3 all visible to set zero position.")
                return

            T_21 = get_21_transform(self.pose1, self.pose2)  # m2 <- m1 (at zero)
            T_31 = get_21_transform(self.pose1, self.pose3)  # m3 <- m1 (at zero)

            self.T_12zero = inverse_transform(T_21)  # m1 <- m2(zero)
            self.T_13zero = inverse_transform(T_31)  # m1 <- m3(zero)
            self.T_plate_last = Orientation(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))

            self.txt_rel.setPlainText("relative (after zero):\nSaved zero position. Move the plate now.")
        except Exception as e:
            self.T_12zero = None
            self.T_13zero = None
            self.T_plate_last = None
            self.txt_rel.setPlainText(f"relative (after zero):\nZeroing error:\n{e}")

    def _plate_motion_from_12(self) -> Optional[Orientation]:
        if self.T_12zero is None or self.pose1 is None or self.pose2 is None:
            return None
        T_2new1 = get_21_transform(self.pose1, self.pose2)  # m2(new) <- m1
        return multiple_transform(T_2new1, self.T_12zero)   # m2(new) <- m2(zero)

    def _plate_motion_from_13(self) -> Optional[Orientation]:
        if self.T_13zero is None or self.pose1 is None or self.pose3 is None:
            return None
        T_3new1 = get_21_transform(self.pose1, self.pose3)  # m3(new) <- m1
        return multiple_transform(T_3new1, self.T_13zero)   # m3(new) <- m3(zero)


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

        if self.quad3 is not None:
            _draw_quad(img, self.quad3, (0, 255, 128), 2)
        if self.pose3 is not None:
            _draw_axes(img, self.pose3, self.K, axis_len=self.len_marker * 0.5)

        self.txt_cam_p1.setPlainText(se3_to_text(self.pose1, "T_cam_m1") if self.pose1 is not None else "T_cam_m1 =\n-")
        self.txt_cam_p2.setPlainText(se3_to_text(self.pose2, "T_cam_m2") if self.pose2 is not None else "T_cam_m2 =\n-")
        self.txt_cam_p3.setPlainText(se3_to_text(self.pose3, "T_cam_m3") if self.pose3 is not None else "T_cam_m3 =\n-")


        if self.T_12zero is None or self.T_13zero is None:
            self.txt_rel.setPlainText("relative (after zero):\nPress Zero button when markers 1,2,3 are visible.")
        else:
            try:
                T22 = self._plate_motion_from_12()
                T33 = self._plate_motion_from_13()
                Tplate, info = pick_or_fuse(T22, T33)
                if Tplate is None:
                    self.txt_rel.setPlainText(f"relative (after zeroing):\n{info}")
                else:
                    self.T_plate_last = Tplate
                    yaw_deg = float(angle_from_transform(Tplate) * 180.0 / np.pi)
                    t = as_col(Tplate.trans).reshape(3)
                    self.txt_rel.setPlainText(
                        f"{info}\n\n"
                        "T_plate:\n"
                        f"{se3_to_text(Tplate, 'T_plate')}\n\n"
                        f"yaw(z) [deg]: {yaw_deg:.2f}\n"
                        f"translation: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"
                    )
            except Exception as e:
                self.txt_rel.setPlainText(f"relative (after zeroing):\nCompute error:\n{e}")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
        except Exception:
            pass


def main():
    CAM_NAME = "frontcam"
    CAM_NPZ_PATH = Path("src") / "camdata" / CAM_NAME / f"{CAM_NAME}.npz"

    try:
        camera = load_camera(CAM_NPZ_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load camera npz: {CAM_NPZ_PATH}\n{e}")

    app = QApplication(sys.argv)
    win = MainWindow(camera=camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
