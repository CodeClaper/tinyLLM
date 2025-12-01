
import tiktoken

class MyTiktoken:
    def __init__(self) -> None:
        self.tokenizer = tiktoken.get_encoding("gpt2")

    ## encode.
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    ## decode.
    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
 
