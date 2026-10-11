from typing import List
from collections import defaultdict
class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        #you wouldjust need to break it done intosomthinge else 
        tokens = list(corpus)
        merged_list = []
        merges = []
        for _ in range(num_merges): 
            key_dict = defaultdict(int)
            for i in range(len(tokens)-1):
                pairs  = (tokens[i], tokens[i+1])
                key_dict[pairs]+=1
            max_value = max(key_dict.values())
            candidates = sorted(p for p,c in key_dict.items() if c ==max_value)
            best = candidates[0]
            merged_list.append([best[0],best[1]])
            new_tokens = []
            i = 0 
            while i <len(tokens): 
                if i <len(tokens) - 1 and tokens[i] == best[0] and tokens[i+1] == best[0+1]: 
                    new_tokens.append(best[0] + best[1])
                    i+=2
                else: 
                    new_tokens.append(tokens[i])
                    i +=1
            tokens = new_tokens
        return merged_list