from CONSTS import METRICS


class Score:
    exact_match: int = 0
    substring: int = 0
    similarity: float = 0.0

    def __getitem__(self, item):
        if item == 'exact_match':
            return self.exact_match
        if item == 'substring':
            return self.substring
        if item == 'similarity':
            return self.similarity

    def __setitem__(self, key, value):
        if key == 'exact_match':
            self.exact_match = value
        if key == 'substring':
            self.substring = value
        if key == 'similarity':
            self.similarity = value

    def __str__(self):
        return f'exact_match: {self.exact_match}; substring: {self.substring}; similarity: {self.similarity:.4f};'

    def to_list(self):
        return [self[k] for k in METRICS]
