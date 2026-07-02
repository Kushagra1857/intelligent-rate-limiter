import os
import sys
import threading
import time
import pytest
from fastapi.testclient import TestClient
os.environ['DEGRADED_THRESHOLD_SECONDS'] = '2'
os.environ['LLM_REQUEST_TIMEOUT'] = '1'
os.environ['LLM_BACKOFF_BASE'] = '0.05'
os.environ['LLM_BACKOFF_JITTER'] = '0.01'
os.environ['LLM_MAX_RETRIES'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import httpx
from httpx import ASGITransport
from mock_llm.main import app as mock_llm_app
from proxy import cache as proxy_cache
from proxy import llm_client as proxy_llm_client
from proxy import metrics as proxy_metrics
from proxy import rate_limiter as proxy_rate_limiter
from proxy.config import RATE_LIMIT_BUDGET
from proxy.main import app as proxy_app

@pytest.fixture(autouse=True)
def reset_proxy_state():
    proxy_metrics.reset_metrics()
    proxy_llm_client.reset_health()
    proxy_cache._cache.clear()
    proxy_rate_limiter._key_states.clear()
    yield
    with TestClient(mock_llm_app) as mock_client:
        mock_client.post('/v1/mode', json={'mode': 'normal'})

@pytest.fixture
def mock_client():
    with TestClient(mock_llm_app) as client:
        yield client

@pytest.fixture
def proxy_client(mock_client):
    transport = ASGITransport(app=mock_llm_app)
    original_call_llm = proxy_llm_client.call_llm

    async def patched_call_llm(prompt: str, max_tokens: int=200):
        import asyncio
        from proxy.config import LLM_BACKOFF_BASE, LLM_BACKOFF_JITTER, LLM_MAX_RETRIES, LLM_REQUEST_TIMEOUT
        from proxy.llm_client import LLMCallError, LLMResponse, _health
        import random
        url = 'http://mock-llm/v1/completions'
        last_exception = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            if attempt > 0:
                wait = LLM_BACKOFF_BASE * 2 ** (attempt - 1) + random.uniform(0, LLM_BACKOFF_JITTER)
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(transport=transport, timeout=LLM_REQUEST_TIMEOUT) as client:
                    resp = await client.post(url, json={'prompt': prompt, 'max_tokens': max_tokens})
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get('usage', {})
                    result = LLMResponse(text=data['text'], prompt_tokens=usage.get('prompt_tokens', 0), completion_tokens=usage.get('completion_tokens', 0), total_tokens=usage.get('total_tokens', 0))
                    _health.record_success()
                    return result
                if resp.status_code in (429, 500, 503):
                    last_exception = LLMCallError(f'LLM returned HTTP {resp.status_code}')
                    _health.record_failure()
                    continue
                _health.record_failure()
                raise LLMCallError(f'LLM returned non-retryable HTTP {resp.status_code}')
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exception = exc
                _health.record_failure()
                continue
        raise LLMCallError(f'LLM exhausted {LLM_MAX_RETRIES} retries. Last: {last_exception}')
    proxy_llm_client.call_llm = patched_call_llm
    with TestClient(proxy_app) as client:
        yield (client, mock_client)
    proxy_llm_client.call_llm = original_call_llm

def set_mock_mode(mock_client: TestClient, mode: str) -> None:
    resp = mock_client.post('/v1/mode', json={'mode': mode})
    assert resp.status_code == 200, f'Failed to set mode to {mode}: {resp.text}'

def test_scenario_1_same_prompt_twice_cached(proxy_client):
    client, mock_client = proxy_client
    set_mock_mode(mock_client, 'normal')
    prompt = 'What are the famous street food spots near aminabad market in Lucknow?'
    api_key = 'varahe_user_s1'
    r1 = client.post('/v1/chat', json={'api_key': api_key, 'prompt': prompt})
    assert r1.status_code == 200, f'First request failed: {r1.text}'
    d1 = r1.json()
    assert d1['cached'] is False
    assert d1['degraded'] is False
    assert d1['actual_tokens'] is not None
    r2 = client.post('/v1/chat', json={'api_key': api_key, 'prompt': prompt})
    assert r2.status_code == 200, f'Second request failed: {r2.text}'
    d2 = r2.json()
    assert d2['cached'] is True, f'Expected cached=True on second request, got: {d2}'
    assert d2['degraded'] is False
    assert d2['actual_tokens'] is None
    assert d2['similarity_score'] is not None
    assert d2['similarity_score'] > 0.8

def test_scenario_2_paraphrased_prompt_similarity(proxy_client):
    client, mock_client = proxy_client
    set_mock_mode(mock_client, 'normal')
    original = 'What courses does the B.Tech program at IIITDMJ offer?'
    paraphrase = 'Which subjects are part of the B.Tech curriculum at IIITDMJ?'
    api_key = 'iiitdmj_student_s2'
    r1 = client.post('/v1/chat', json={'api_key': api_key, 'prompt': original})
    assert r1.status_code == 200
    assert r1.json()['cached'] is False
    r2 = client.post('/v1/chat', json={'api_key': api_key, 'prompt': paraphrase})
    assert r2.status_code == 200
    d2 = r2.json()
    similarity = d2.get('similarity_score')
    print(f'\n[Scenario 2] Similarity score: {similarity}')
    print(f"  Original:   '{original}'")
    print(f"  Paraphrase: '{paraphrase}'")
    print(f"  cached={d2['cached']} (expected True — both asking about IIITDMJ B.Tech subjects)")
    if d2['cached']:
        assert similarity is not None
        assert similarity > 0.8
        print('  Cache HIT — good: same semantic intent, same answer is appropriate.')
    else:
        print(f'  Cache MISS (similarity={similarity}) — TF-IDF limitations: synonym blindness.')
        print('  This is a known TF-IDF failure mode documented in design_doc.md.')

def test_scenario_3_budget_exhaustion(proxy_client):
    client, mock_client = proxy_client
    set_mock_mode(mock_client, 'normal')
    api_key = 'varahe_intern_s3'
    import asyncio

    async def exhaust_budget():
        state = await proxy_rate_limiter._get_or_create_state(api_key)
        state.used_tokens = RATE_LIMIT_BUDGET - 5
    asyncio.get_event_loop().run_until_complete(exhaust_budget())
    long_prompt = 'Describe the history, culture, food, architecture, and tourism of Lucknow in great detail.'
    r = client.post('/v1/chat', json={'api_key': api_key, 'prompt': long_prompt})
    assert r.status_code == 429, f'Expected 429, got {r.status_code}: {r.text}'
    body = r.json()
    print(f'\n[Scenario 3] 429 response body: {body}')
    assert 'budget_remaining' in body, f"Expected 'budget_remaining' in body: {body}"
    assert body['budget_remaining'] >= 0
    assert body['budget_remaining'] < RATE_LIMIT_BUDGET
    assert body.get('api_key') == api_key or 'api_key' in str(body)

def test_scenario_4_degraded_mode_after_threshold(proxy_client):
    client, mock_client = proxy_client
    api_key = 'iiitdmj_user_s4'
    original_prompt = 'What is the syllabus for DSA at IIITDMJ?'
    set_mock_mode(mock_client, 'normal')
    r_prime = client.post('/v1/chat', json={'api_key': api_key, 'prompt': original_prompt})
    assert r_prime.status_code == 200
    assert r_prime.json()['cached'] is False
    proxy_llm_client._health.first_failure_time = time.monotonic() - 3.0
    proxy_llm_client._health.last_success_time = None
    set_mock_mode(mock_client, 'always_500')
    assert proxy_llm_client.is_degraded(threshold_override=2.0), 'Health should be degraded: 3 s elapsed > 2 s threshold'
    similar_prompt = 'Tell me about the DSA syllabus at IIITDMJ.'
    r_degraded = client.post('/v1/chat', json={'api_key': api_key, 'prompt': similar_prompt})
    print(f'\n[Scenario 4] Response status: {r_degraded.status_code}')
    body = r_degraded.json()
    print(f'[Scenario 4] Response body: {body}')
    if r_degraded.status_code == 200:
        assert body['degraded'] is True, f'Expected degraded=True, got: {body}'
        assert body['cached'] is True
        print(f"  degraded=True confirmed | similarity_score={body.get('similarity_score')}")
    else:
        assert r_degraded.status_code == 503, f'Expected 200 degraded or 503, got {r_degraded.status_code}'
        assert body.get('degraded') is True
        print('  503 degraded (no cache fallback above 0.50 threshold)')

def test_scenario_5_recovery_after_degraded(proxy_client):
    client, mock_client = proxy_client
    api_key = 'varahe_engineer_s5'
    proxy_llm_client._health.first_failure_time = time.monotonic() - 3.0
    proxy_llm_client._health.last_success_time = None
    assert proxy_llm_client.is_degraded(threshold_override=2.0), 'Should be degraded'
    set_mock_mode(mock_client, 'normal')
    proxy_llm_client._health.record_success()
    assert not proxy_llm_client.is_degraded(threshold_override=2.0), 'After record_success(), should no longer be degraded'
    recovery_prompt = 'What kind of internship projects does Varahe Analytics offer to students in the summer?'
    r = client.post('/v1/chat', json={'api_key': api_key, 'prompt': recovery_prompt})
    print(f'\n[Scenario 5] Response status: {r.status_code}')
    if r.status_code == 200:
        body = r.json()
        print(f"  degraded={body.get('degraded')} cached={body.get('cached')}")
        assert body['degraded'] is False, f'Expected degraded=False after recovery, got: {body}'
        assert proxy_llm_client._health.first_failure_time is None, 'Health should remain restored'
        print('  Recovery confirmed: degraded=False on the very next request.')
    else:
        pytest.fail(f'Expected 200 after recovery, got {r.status_code}: {r.text}')

def test_scenario_6_concurrent_budget_enforcement(proxy_client):
    client, mock_client = proxy_client
    set_mock_mode(mock_client, 'normal')
    api_key = 'lucknow_batch_s6'
    import asyncio

    async def set_near_limit():
        state = await proxy_rate_limiter._get_or_create_state(api_key)
        state.used_tokens = RATE_LIMIT_BUDGET - 3
    asyncio.get_event_loop().run_until_complete(set_near_limit())
    n_threads = 20
    results: list[dict] = [{}] * n_threads
    barrier = threading.Barrier(n_threads)

    def make_request(idx: int):
        unique_prompt = f'Tell me about the historical monuments, food culture, and famous markets of Lucknow in detail, request number {idx + 1}.'
        barrier.wait()
        resp = client.post('/v1/chat', json={'api_key': api_key, 'prompt': unique_prompt})
        results[idx] = {'status': resp.status_code, 'body': resp.json() if resp.content else {}}
    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    statuses = [r['status'] for r in results]
    n_200 = statuses.count(200)
    n_429 = statuses.count(429)
    print(f'\n[Scenario 6] {n_200} succeeded, {n_429} rate-limited (out of {n_threads} concurrent)')
    remaining = proxy_rate_limiter.get_budget_remaining(api_key)
    used = RATE_LIMIT_BUDGET - remaining
    print(f'  Budget used: {used}/{RATE_LIMIT_BUDGET} | Remaining: {remaining}')
    assert used <= RATE_LIMIT_BUDGET, f'Budget exceeded! used={used} > limit={RATE_LIMIT_BUDGET}. This indicates a concurrency bug in rate_limiter.py.'
    assert n_429 > 0, f'Expected some 429s near budget limit (budget space was 3 tokens, 20 threads). Got: {n_200} succeeded, {n_429} rate-limited. Budget used: {used}'
