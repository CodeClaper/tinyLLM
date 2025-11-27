## This module aims to tokenize text.

import re

## tokenize text
def tokenize_text(text: str) -> list[str]:
    result = re.split(r'([.,:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]

## tokenize file.
def tokenize_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return tokenize_text(raw_text)


