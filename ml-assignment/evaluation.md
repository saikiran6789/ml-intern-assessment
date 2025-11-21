
Evaluation Summary – Trigram Language Model

This document summarizes the design choices behind the implementation of the TrigramModel for the AI/ML Internship Assignment.

1. Data Structure for N-gram Counts

I used the following structure:

self.trigram_counts = defaultdict(lambda: defaultdict(int))


Reasoning:

A trigram represents (w1, w2) → w3.

Using a nested defaultdict allows automatic initialization of counters.

It avoids manual key checks and keeps the code clean.

Efficient for counting and retrieving probability distributions.

This structure stores counts like:

("the", "big") → {"dog": 3, "house": 1}

2. Text Cleaning & Tokenization

Text preprocessing pipeline:

Lowercasing
Ensures uniformity and reduces vocabulary size.

Removing punctuation
Done with a regex to keep only alphanumeric characters and spaces.

Tokenization
Split text using simple whitespace-based .split().

Reasoning:
For trigram modeling, simple tokenization is enough. More complex tokenizers were unnecessary for this task.

3. Start/End Padding

Before n-gram extraction, the token list is padded:

["<s>", "<s>", ...tokens..., "</s>"]


Why this is important:

Allows the model to generate a meaningful beginning of a sentence.

The end token "</s>" provides a natural stopping point during generation.

4. Handling Unknown Words

Unknown words are handled implicitly:

During training, every word seen is added to a set called self.vocab.

If during generation a context (w1, w2) has no trigram entries, the model falls back to:

random.choice(list(self.vocab))


This prevents the model from getting stuck.

5. Probability Sampling in generate()

To choose the next word, I use weighted random sampling:

random.choices(words, probs)[0]


Where:

words = all possible continuations of (w1, w2)

probs = count(w3) / total count

Why:
random.choices provides a clean and simple way to sample based on probabilities without manually computing cumulative distributions.

6. Text Generation Logic

Algorithm:

Start with w1 = "<s>" and w2 = "<s>".

Sample the next word from the trigram distribution.

Append it to output unless it is "</s>".

Slide the window:

(w1, w2) = (w2, w3)


Continue until:

The end token appears, or

The max_length is reached.

Design Reasoning:
This closely mimics the natural generative behavior of a trigram model while preventing infinite loops.

7. Other Considerations

Used defaultdict(int) to simplify counting logic.

Avoided external libraries to keep implementation clean and fully compliant with assignment expectations.

Ensured deterministic structure with probabilistic sampling for realistic text generation.

Code passes all tests provided in the assignment.

Conclusion

The final trigram model is clean, maintainable, and aligns with standard NLP n-gram modeling practices. The design choices prioritize simplicity, readability, and testability while meeting all functional requirements of the assignment.
