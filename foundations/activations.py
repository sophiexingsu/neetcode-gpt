import numpy as np
from numpy.typing import NDArray
import math
class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        answer = []
        for d in z: 
            sig_value = 1/(1+ math.exp(-d))
            answer.append(round(sig_value, 5))
        return answer
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        answer = []
        for d in z: 
            relu = max(0.0, d)
            answer.append(relu)
        return answer
        pass
