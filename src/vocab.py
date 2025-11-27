## This module aims to generate vocabulary.

from src.tokenize import tokenize

class vocab:
    def __init__(self) -> None:
        self.tokenize_instance = tokenize()

    ## vocab text.
    def vocab_text(self, text: str) -> dict[str, int]:
        tokens = self.tokenize_instance.tokenize_text(text)
        all_words = sorted(set(tokens))
        return {token: integer for integer, token in enumerate(all_words)}

    ## vocab file.
    def vacab_file(self, file_path: str) -> dict[str, int]:
        tokens = self.tokenize_instance.tokenize_file(file_path)
        all_words = sorted(set(tokens))
        return {token: integer for integer, token in enumerate(all_words)}
