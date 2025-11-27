# This is test for tokenize module.

from src.tokenize import tokenize
from src.tokenize import tokenize

tokenize_instance = tokenize()

## test for function <tokenize_text>
def test_token_text():
    ret = tokenize_instance.tokenize_text("Hello world, this is a test.")
    assert ret == ['Hello', 'world', ',', 'this', 'is', 'a', 'test', "."]

## test for function <tokenize_file>
def test_token_file():
    ret = tokenize_instance.tokenize_file("./the-verdict.txt")
    assert ret[:20] == ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was'] 

