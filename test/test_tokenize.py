# This is test for tokenize module.

from src.tokenize import tokenize_text
from src.tokenize import tokenize_file

## test for function <tokenize_text>
def test_token_text():
    ret = tokenize_text("Hello world, this is a test.")
    assert ret == ['Hello', 'world', ',', 'this', 'is', 'a', 'test']

## test for function <tokenize_file>
def test_token_file():
    ret = tokenize_file("./the-verdict.txt")
    assert ret[:20] == [] 

