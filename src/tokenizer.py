from tokenizers import Tokenizer

class ParakeetTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str) -> 'ParakeetTokenizer':
        # Load tokenizer.json from path
        tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        return cls(tokenizer)

    def __getattr__(self, name):
        # Forward any undefined attribute access to the underlying tokenizer
        return getattr(self.tokenizer, name)