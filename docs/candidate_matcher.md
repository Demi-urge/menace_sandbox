# Candidate Matcher

`find_matching_models` compares a `NicheCandidate` against stored models and returns the best matches.

Text similarity normally relies on `sklearn`'s `TfidfVectorizer`. When that package
is missing, a lightweight TF‑IDF fallback is used. The fallback now removes common
stop words, weights tokens by length and keeps a growing corpus for calculating
inverse document frequency. Cosine similarity is computed directly from these
vectors so model matching works even without external libraries.
