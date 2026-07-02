import pytest
from proxy.similarity import cosine_similarity, inverse_document_frequency, most_similar, term_frequency, tfidf_vectors, tokenize

class TestTokenize:

    def test_basic(self) -> None:
        assert tokenize('Hello, World!') == ['hello', 'world']

    def test_empty(self) -> None:
        assert tokenize('') == []

    def test_numbers(self) -> None:
        tokens = tokenize('version 3.14')
        assert 'version' in tokens
        assert '3' in tokens or '14' in tokens

    def test_uppercase_normalized(self) -> None:
        assert tokenize('PYTHON') == ['python']

    def test_punctuation_stripped(self) -> None:
        result = tokenize('one,two;three')
        assert result == ['one', 'two', 'three']

class TestTermFrequency:

    def test_equal_words(self) -> None:
        tf = term_frequency(['a', 'b', 'c'])
        assert abs(tf['a'] - 1 / 3) < 1e-09
        assert abs(tf['b'] - 1 / 3) < 1e-09

    def test_repeated_word(self) -> None:
        tf = term_frequency(['cat', 'cat', 'dog'])
        assert abs(tf['cat'] - 2 / 3) < 1e-09
        assert abs(tf['dog'] - 1 / 3) < 1e-09

    def test_empty_tokens(self) -> None:
        assert term_frequency([]) == {}

class TestIDF:

    def test_word_in_all_docs(self) -> None:
        idf = inverse_document_frequency([['a', 'b'], ['a', 'c']], ['a', 'b', 'c'])
        assert abs(idf['a'] - 1.0) < 1e-09

    def test_word_in_one_doc(self) -> None:
        import math
        idf = inverse_document_frequency([['a', 'b'], ['a', 'c']], ['a', 'b', 'c'])
        expected = math.log(3 / 2) + 1.0
        assert abs(idf['b'] - expected) < 1e-09

    def test_empty_corpus(self) -> None:
        idf = inverse_document_frequency([], ['a', 'b'])
        assert idf['a'] == 1.0
        assert idf['b'] == 1.0

class TestCosineSimilarity:

    def test_identical_vectors(self) -> None:
        vec = {'a': 1.0, 'b': 2.0}
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-09

    def test_orthogonal_vectors(self) -> None:
        a = {'x': 1.0}
        b = {'y': 1.0}
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vector(self) -> None:
        assert cosine_similarity({}, {'a': 1.0}) == 0.0
        assert cosine_similarity({'a': 1.0}, {}) == 0.0
        assert cosine_similarity({}, {}) == 0.0
NEAR_DUPLICATE_A = 'What are the famous street food spots near Aminabad market in Lucknow?'
NEAR_DUPLICATE_B = 'What are the famous street food places near Aminabad market in Lucknow?'
RELATED_A = 'What courses does the B.Tech program at IIITDMJ offer?'
RELATED_B = 'Which subjects are part of the B.Tech curriculum at IIITDMJ?'
UNRELATED_A = 'What are the famous street food spots near Aminabad in Lucknow?'
UNRELATED_B = 'What kind of internship projects does Varahe Analytics offer to students in the summer?'

def _compute_pair_similarity(doc_a: str, doc_b: str) -> float:
    vectors = tfidf_vectors([doc_a, doc_b])
    return cosine_similarity(vectors[0], vectors[1])

class TestRequiredPairs:

    def test_near_duplicate_above_threshold(self) -> None:
        score = _compute_pair_similarity(NEAR_DUPLICATE_A, NEAR_DUPLICATE_B)
        print(f"\n[near-duplicate] '{NEAR_DUPLICATE_A}' vs '{NEAR_DUPLICATE_B}' -> score={score:.4f}")
        assert score > 0.8, f'Expected > 0.80, got {score:.4f}'

    def test_related_below_threshold(self) -> None:
        score = _compute_pair_similarity(RELATED_A, RELATED_B)
        print(f"\n[related-distinct] '{RELATED_A}' vs '{RELATED_B}' -> score={score:.4f}")
        assert score < 0.8, f'Expected < 0.80, got {score:.4f}'

    def test_unrelated_near_zero(self) -> None:
        score = _compute_pair_similarity(UNRELATED_A, UNRELATED_B)
        print(f"\n[unrelated] '{UNRELATED_A}' vs '{UNRELATED_B}' -> score={score:.4f}")
        assert score < 0.15, f'Expected near 0.0, got {score:.4f}'

def test_print_similarity_pairs(capsys) -> None:
    pairs = [('near-duplicate', NEAR_DUPLICATE_A, NEAR_DUPLICATE_B, '> 0.80'), ('related-distinct', RELATED_A, RELATED_B, '< 0.80'), ('unrelated', UNRELATED_A, UNRELATED_B, '~= 0.0')]
    with capsys.disabled():
        print('\n\n=== TF-IDF Cosine Similarity - Required Pairs ===')
        for label, a, b, expected in pairs:
            score = _compute_pair_similarity(a, b)
            print(f'[{label}]')
            print(f'  A: {a!r}')
            print(f'  B: {b!r}')
            print(f'  Score: {score:.4f}  (expected {expected})')
            print()

class TestMostSimilar:

    def test_empty_candidates(self) -> None:
        score, idx = most_similar('hello', [])
        assert score == 0.0
        assert idx is None

    def test_exact_match(self) -> None:
        score, idx = most_similar('What is the DSA syllabus at IIITDMJ?', ['What is the DSA syllabus at IIITDMJ?', 'something else'])
        assert idx is not None

    def test_best_candidate_selected(self) -> None:
        query = 'What is the syllabus for DSA at IIITDMJ?'
        candidates = ['Famous food spots near Aminabad in Lucknow.', 'Tell me about the DSA syllabus at IIITDMJ.', 'What kind of internships does Varahe Analytics offer?']
        score, idx = most_similar(query, candidates)
        assert idx == 1, f'Expected index 1 (DSA syllabus), got {idx}'
        assert score > 0.0
