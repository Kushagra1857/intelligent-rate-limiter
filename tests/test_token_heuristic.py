import re
import pytest
from proxy.token_heuristic import estimate_tokens
_GT_SPLIT_PATTERN = re.compile('[\\s\\.,;:!?(){}\\[\\]\\"\'`<>|/\\\\@#$%^&*+=~\\-]+')

def ground_truth_count(text: str) -> int:
    pieces = _GT_SPLIT_PATTERN.split(text)
    return sum((1 for p in pieces if p))
SAMPLE_PROMPTS = [('simple_prose', 'The famous Nawabi cuisine of Lucknow is known all over India for its unique taste.'), ('question_prose', 'What is the placement process at IIIT DM Jabalpur and which companies visit every year?'), ('python_function', 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)'), ('json_object', '{"name": "Kushal", "city": "Lucknow", "college": "IIITDMJ", "intern": true}'), ('nested_json', '{"company":{"name":"Varahe Analytics","teams":[{"id":1,"name":"backend"},{"id":2,"name":"ml"}]},"headcount":12}'), ('mixed_prose_code', "Call the proxy like this: requests.post('http://localhost:8000/v1/chat', json={'api_key': 'varahe_intern', 'prompt': 'Explain rate limiting'})"), ('url_only', 'http://localhost:8000/v1/metrics?api_key=varahe_user_1&format=json#budget'), ('empty_string', ''), ('whitespace_only', '     \n\t   '), ('long_single_token', 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')]

@pytest.mark.parametrize('label,text', SAMPLE_PROMPTS)
def test_estimate_vs_ground_truth(label: str, text: str) -> None:
    gt = ground_truth_count(text)
    est = estimate_tokens(text)
    if label == 'empty_string':
        assert est == 0, f'empty string should give 0, got {est}'
        return
    if label == 'whitespace_only':
        assert est == 0, f'whitespace-only string should give 0, got {est}'
        return
    if label == 'long_single_token':
        assert est == 1, f"long single 'word' should give 1, got {est}"
        return
    if gt == 0:
        assert est == 0
        return
    error_pct = abs(est - gt) / gt * 100
    assert error_pct <= 25.0, f'[{label}] estimate={est} ground_truth={gt} error={error_pct:.1f}% (>25% tolerance)'

def test_print_validation_table(capsys) -> None:
    rows = []
    for label, text in SAMPLE_PROMPTS:
        gt = ground_truth_count(text)
        est = estimate_tokens(text)
        if gt > 0:
            err = abs(est - gt) / gt * 100
            err_str = f'{err:.1f}%'
        elif est == 0:
            err_str = '0.0% (both 0)'
        else:
            err_str = 'N/A (gt=0)'
        rows.append((label, gt, est, err_str))
    with capsys.disabled():
        print('\n\n=== Token Estimation Validation Table ===')
        print(f"{'Prompt Label':<25} {'Ground Truth':>12} {'Estimated':>10} {'Error %':>10}")
        print('-' * 62)
        for label, gt, est, err in rows:
            print(f'{label:<25} {gt:>12} {est:>10} {err:>10}')
        print('=' * 62)

def test_empty_string() -> None:
    assert estimate_tokens('') == 0

def test_whitespace_only() -> None:
    assert estimate_tokens('   \n\t  ') == 0

def test_simple_words() -> None:
    assert estimate_tokens('Lucknow India') == 2

def test_punctuation_dense_code() -> None:
    result = estimate_tokens("student = {'name': 'Kushal', 'college': 'IIITDMJ'}")
    assert result >= 5, f'expected >= 5 tokens for code, got {result}'

def test_single_long_word() -> None:
    url = 'http://localhost:8000/v1/chat?api_key=varahe_user_1&session=abc123xyz&format=json'
    result = estimate_tokens(url)
    assert result >= 1

def test_deeply_nested_json() -> None:
    text = '{"varahe":{"team":{"member":{"name":"Kushal","role":"intern"}}}}'
    result = estimate_tokens(text)
    assert result >= 5
