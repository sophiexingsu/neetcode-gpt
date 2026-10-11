import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5) 
        #I think right now, there is just one layersa nd an activaiton layer and the #han an output layers
        #first layer weights = weigths[0]
        layer = x
        for i in range(len(weights)-1): 
            layer = layer @ weights[i] + biases[i]
            layer = np.maximum(layer, 0)
        output_layer = layer @ weights[-1] + biases[-1]
        return np.round(output_layer,5)
