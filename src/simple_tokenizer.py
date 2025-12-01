## This module is a simple tokenizer supports <encode> and <decode> meothods.

import re
from src.tokenize import Tokenize

class SimpleTokenizer:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = { i : s for s,i in vocab.items()}
        self.tokenize_instance = Tokenize()

    ## Encode text to number array.
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize_instance.tokenize_text(text)
        ids = [self.str_to_int[s] for s in tokens]
        return ids
    
    ## Decode number array to text.
    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

