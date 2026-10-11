import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        w1 = np.array(W1)
        w2 = np.array(W2)
        b1=  np.array(b1)
        b2 = np.array(b2)
        n = len(x)
        y_true = np.array(y_true)
        #first layer outpuot
        first_layer = np.dot(x, w1.T) + b1 #pre-activation layer1
        relu_layer = np.maximum(first_layer,0.0) #Relu Activation
        second_layer = np.dot(relu_layer, w2.T) + b2 #output 
        loss =  np.mean((y_true - second_layer)**2)
        #backward pass
        n = len(y_true)  if y_true.ndim > 0 else 1
        grad = 2*(second_layer-y_true)/n #thsi is the mor eimproant theitng
        dw2 = np.outer(grad, relu_layer)
        db2 = grad
        da1 = grad @ w2 #this is the backpropogations
        dz1 = da1 * (relu_layer > 0)
        dw1 = np.outer(dz1, x)
        db1 = dz1
        return {
            "loss": np.round(loss,4), 
            "dW1": np.round(dw1,4), 
            "db1": np.round(dz1,4), 
            "dW2": np.round(dw2,4), 
            "db2": np.round(grad,4)
        }
        pass
