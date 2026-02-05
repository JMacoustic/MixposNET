import numpy as np
from dataclasses import dataclass

class QRtransform:
    def __init__(self, rotation: np.ndarray = np.identity(3), translation: np.ndarray = np.zeros((3, 1))):
        self.rot = rotation
        self.trans = translation
    
    def reset(self):
        self.rot = np.identity(3)
        self.trans = np.zeros((3, 1))
  

@dataclass(frozen=True)
class QRdata:
    decoded_qr: str #= "Empty"
    center_pos: np.ndarray #= np.zeros(2)
    corner_pos: np.ndarray #= np.zeros((4, 2))
    orientation: QRtransform #= QRtransform(np.identity(3), np.zeros((3, 1)))

    def __str__(self) -> str:
        def fmt(arr: np.ndarray) -> str:
            return np.array2string(
                arr,
                precision=4,
                suppress_small=True,
                separator=", "
            )

        return (
            "\n"
            f"========== {self.decoded_qr} ==========\n"
            f"  center : {fmt(self.center_pos)},\n"
            f"  corners :\n{fmt(self.corner_pos)},\n"
            f"  rotation :\n{fmt(self.orientation.rot)},\n"
            f"  translation :\n{fmt(self.orientation.trans)}\n"
        )


def inverse_transform(T: QRtransform):
    R_mat = T.rot
    P_vec = T.trans

    R_inv = R_mat.T
    P_inv = - np.matmul(R_inv, P_vec)

    return QRtransform(R_inv, P_inv)


def multiple_transform(T1: QRtransform, T2: QRtransform):
    R_mat1 = T1.rot
    P_vec1 = T1.trans
    R_mat2 = T2.rot
    P_vec2 = T2.trans

    R_mul = np.matmul(R_mat1, R_mat2)
    P_mul = np.matmul(R_mat1, P_vec2) + P_vec1

    return QRtransform(R_mul, P_mul)