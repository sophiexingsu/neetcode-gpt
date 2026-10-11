from typing import Dict, List, Tuple
from collections import defaultdict
class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        letter_set = set()
        for s in text:
            letter_set.add(s)
        letter_set = sorted(letter_set)
        stoi = defaultdict(int)
        itos = defaultdict(str)
        for i in range(len(letter_set)): 
            stoi[letter_set[i]] =i
            itos[i] = letter_set[i]
        return (stoi, itos)
        pass

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encode_list = []
        for s in text: 
            encode_list.append(stoi[s])
        return encode_list
        pass

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decode_string = ""
        for i in ids: 
            decode_string += itos[i]
        return decode_string
        pass
