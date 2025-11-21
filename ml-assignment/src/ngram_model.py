import random
import re
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # trigram_counts[(w1, w2)][w3] = count
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def _clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = text.split()
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.
        """
        tokens = self._clean_and_tokenize(text)

        # Add padding start + end tokens
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]

        self.vocab.update(tokens)

        # Count trigrams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigram_counts[(w1, w2)][w3] += 1

    def _choose_next_word(self, w1, w2):
        options = self.trigram_counts.get((w1, w2), {})
        if not options:
            # If no trigram exists, pick any word (fallback)
            return random.choice(list(self.vocab))

        words = list(options.keys())
        counts = list(options.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        return random.choices(words, probs)[0]

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.
        """
        w1, w2 = "<s>", "<s>"
        result = []

        for _ in range(max_length):
            w3 = self._choose_next_word(w1, w2)
            if w3 == "</s>":
                break
            result.append(w3)
            w1, w2 = w2, w3

        return " ".join(result)
