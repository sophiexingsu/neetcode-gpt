import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        b = 0 
        n_features = len(X[0]) 
        n_samples = len(X)
        w = np.zeros(n_features)
        for epoch in range(epochs): 
            y_hat = X@ w + b
            grad_w = 2/n_samples*(X.T@(y_hat - y))
            grad_b = 2/n_samples*np.sum(y_hat - y)
            w -= lr*grad_w
            b -= lr*grad_b
        return (np.round(w,5), np.around(b,5))
