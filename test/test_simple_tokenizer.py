# This is test for SimpleTokenizer module.

from src.simple_tokenizer import SimpleTokenizer
from src.vocab import Vocab

vocab_instance = Vocab()
vocab = vocab_instance.vocab_file("./the-verdict.txt")
simple_tokenizer_instance = SimpleTokenizer(vocab)

## test for encode.
def test_encode():
    text = """"It's the last he said, you know," Mrs. Gisburn said with pardonable pride."""
    ids = simple_tokenizer_instance.encode(text)
    assert ids == [1, 56, 2, 850, 988, 602, 533, 851, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]


## test for decode.
def test_decode():
    ids = [1, 56, 2, 850, 988, 602, 533, 851, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
    text = simple_tokenizer_instance.decode(ids)
    assert text == '" It\' s the last he said, you know," Mrs. Gisburn said with pardonable pride.'

## test raise key error.
def test_unk_token():
    text = "Hello, do you like tea?"
    ids = simple_tokenizer_instance.encode(text)
    assert ids == [1131, 5, 355, 1126, 628, 975, 10]
