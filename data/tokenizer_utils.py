from typing import List, Dict
import numpy as np
class Solution:
    def  _greedy_tokenizer(self, text, vocab): 
        tokens = []
        i = 0
        while i < len(text): 
            best =None
            for length in range(len(text)-i, 0, -1): 
                substr = text[i:i +length]
                if substr in vocab: 
                    best = substr
                    break
            if not best: 
                tokens.append(text[i])
                i+=1
            else: 
                tokens.append(best)
                i+= len(best)
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        all_tokened = []
        for number in numbers:
            number_tokened = self._greedy_tokenizer(str(number), vocab)
            all_tokened.append(number_tokened)
        return all_tokened
        pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens =  self._greedy_tokenizer(text, vocab)
        count  = len(tokens)
        return count
        pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        counts = self.count_tokens(text, vocab)
        words = text.split()
        ratio = np.round(counts/len(words),4)
        return ratio
