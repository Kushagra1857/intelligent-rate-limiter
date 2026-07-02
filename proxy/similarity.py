import math
import re
from collections import Counter
from typing import Optional

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall('\\w+', text.lower())

def term_frequency(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    total = len(tokens)
    counts = Counter(tokens)
    return {word: count / total for word, count in counts.items()}

def inverse_document_frequency(token_lists: list[list[str]], vocab: list[str]) -> dict[str, float]:
    n = len(token_lists)
    if n == 0:
        return {word: 1.0 for word in vocab}
    doc_sets = [set(tokens) for tokens in token_lists]
    idf: dict[str, float] = {}
    for word in vocab:
        df = sum((1 for doc_set in doc_sets if word in doc_set))
        idf[word] = math.log((n + 1) / (df + 1)) + 1.0
    return idf

def tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    if not documents:
        return []
    token_lists = [tokenize(doc) for doc in documents]
    vocab = list({word for tokens in token_lists for word in tokens})
    idf = inverse_document_frequency(token_lists, vocab)
    tfs = [term_frequency(tokens) for tokens in token_lists]
    vectors: list[dict[str, float]] = []
    for tf in tfs:
        vec: dict[str, float] = {}
        for word in vocab:
            score = tf.get(word, 0.0) * idf.get(word, 0.0)
            if score != 0.0:
                vec[word] = score
        vectors.append(vec)
    return vectors

def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum((a.get(word, 0.0) * b.get(word, 0.0) for word in b))
    mag_a = math.sqrt(sum((v * v for v in a.values())))
    mag_b = math.sqrt(sum((v * v for v in b.values())))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)

def most_similar(query: str, candidates: list[str]) -> tuple[float, Optional[int]]:
    if not candidates:
        return (0.0, None)
    all_docs = [query] + candidates
    vectors = tfidf_vectors(all_docs)
    query_vec = vectors[0]
    candidate_vecs = vectors[1:]
    best_score = 0.0
    best_idx: Optional[int] = None
    for idx, vec in enumerate(candidate_vecs):
        score = cosine_similarity(query_vec, vec)
        if score > best_score:
            best_score = score
            best_idx = idx
    return (best_score, best_idx)
