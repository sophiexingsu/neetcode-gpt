import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        x_data =[]
        y_data = []
        raw_list = raw_dataset.split()
        random_indexes =torch.randint(0, len(raw_list)- context_length, size = (batch_size,))
        for index in random_indexes: 
            x_instance = raw_list[index: index+context_length]
            y_instance = raw_list[index+1: index+context_length+1]
            x_data.append(x_instance)
            y_data.append(y_instance)
        #X = torch.stack(x_data)
        #Y = torch.stack(y_data)
        return x_data, y_data
        pass
