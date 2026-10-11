import numpy as np
from numpy.typing import NDArray
import math
class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_value = np.max(z)
        z -= max_value
        logit = [math.exp(i) for i in z]
        logit_sum = np.sum(logit)
        softmax = [np.round(x/logit_sum, 4) for x in logit]
        return softmax
        pass
