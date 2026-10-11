import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        pred_y = 1 / (1 + np.exp(-z))

        n_features = len(w)
        dl_dw = []

        for i in range(n_features):
            dl_dwi = (pred_y - y_true) * pred_y * (1 - pred_y) * x[i]
            dl_dw.append(dl_dwi)

        dl_db = (pred_y - y_true) * pred_y * (1 - pred_y)

        return (np.round(dl_dw, 5), np.round(dl_db, 5))