class CharacterTokenizer:

    def __init__(self, text):

        self.chars = sorted(set(text))

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):

        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, token_ids):

        return "".join(
            self.itos[i]
            for i in token_ids
        )