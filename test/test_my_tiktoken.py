## Test for modle tiktoken

from src.my_tiktoken import MyTiktoken

instance = MyTiktoken()

## test for encode.
def test_encode():
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    ids = instance.encode(text)
    assert ids == [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
