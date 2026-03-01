# tokenizer.py
class ChessTokenizer:
    def __init__(self,
                 chars =  "abcdefgh0123456789prnqkPRNBQK/- w"):
                 
        self.char_to_int = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        self.int_to_char = {0: "<PAD>", 1: "<START>", 2: "<END>"}
        
        for i, char in enumerate(chars):
            self.char_to_int[char] = i + 3
            self.int_to_char[i + 3] = char
            
        self.vocab_size = len(self.char_to_int)

    def encode(self, text, is_target=False):
        tokens = [self.char_to_int[c] for c in text if c in self.char_to_int]
        if is_target:
            tokens = [self.char_to_int["<START>"]] + tokens + [self.char_to_int["<END>"]]
        return tokens

    def decode(self, tokens):
        chars = []
        for token in tokens:
            if token in (0, 1): continue
            if token == 2: break
            chars.append(self.int_to_char.get(token, ""))
        return "".join(chars)