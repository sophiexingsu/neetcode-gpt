import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)
        start_indices = torch.randint(
            0, len(data)-context_length, size = (batch_size, )
        )
        x_data =[]
        y_data = []
        for start_index in start_indices: 
            x_data_instance = data[start_index: start_index + context_length]
            y_data_instance = data[start_index+1: start_index+1+context_length]
            x_data.append(x_data_instance)
            y_data.append(y_data_instance)
        X = torch.stack(x_data)
        Y = torch.stack(y_data)
        return (X,Y)
        pass
