## This module aims to tokenize text.

import re

class tokenize:
    def __init__(self) -> None:
        pass

    ## tokenize text
    def tokenize_text(self, text: str) -> list[str]:
        result = re.split(r'([.,:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in result if item.strip()]

    ## tokenize file.
    def tokenize_file(self, file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        return self.tokenize_text(raw_text)


