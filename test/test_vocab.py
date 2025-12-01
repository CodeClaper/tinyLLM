# This is test for vacab module.

from src.vocab import Vocab

vocab_instance = Vocab()

## test for function <tokenize_text>
def test_token_text():
    ret = vocab_instance.vocab_text("Hello world, this is a test.")
    assert ret == {',': 0, '.': 1, 'Hello': 2, 'a': 3, 'is': 4, 'test': 5, 'this': 6, 'world': 7}

## test for function <tokenize_file>
def test_token_file():
    ret = vocab_instance.vocab_file("./the-verdict.txt")
    assert sorted(ret.items())[:20] == [('!', 0), ('"', 1), ("'", 2), ('(', 3), (')', 4), (',', 5), ('--', 6), ('.', 7), (':', 8), (';', 9), ('?', 10), ('A', 11), ('Ah', 12), ('Among', 13), ('And', 14), ('Are', 15), ('Arrt', 16), ('As', 17), ('At', 18), ('Be', 19)]

