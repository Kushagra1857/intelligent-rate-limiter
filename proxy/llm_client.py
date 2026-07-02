import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx
from proxy.config import DEGRADED_THRESHOLD_SECONDS, LLM_BACKOFF_BASE, LLM_BACKOFF_JITTER, LLM_MAX_RETRIES, LLM_REQUEST_TIMEOUT, MOCK_LLM_BASE_URL
logger = logging.getLogger('proxy.llm_client')

@dataclass
class _HealthState:
    first_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

    def record_success(self) -> None:
        self.first_failure_time = None
        self.last_success_time = time.monotonic()

    def record_failure(self) -> None:
        if self.first_failure_time is None:
            self.first_failure_time = time.monotonic()

    def is_degraded(self, threshold: float=DEGRADED_THRESHOLD_SECONDS) -> bool:
        if self.first_failure_time is None:
            return False
        elapsed = time.monotonic() - self.first_failure_time
        return elapsed >= threshold
_health = _HealthState()

class LLMCallError(Exception):
    pass

class DegradedNoFallbackError(Exception):
    pass

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

async def call_llm(prompt: str, max_tokens: int=200) -> LLMResponse:
    url = f'{MOCK_LLM_BASE_URL}/v1/completions'
    last_exception: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        if attempt > 0:
            wait = LLM_BACKOFF_BASE * 2 ** (attempt - 1) + random.uniform(0, LLM_BACKOFF_JITTER)
            logger.info('retry attempt=%d | waiting=%.2fs', attempt, wait)
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
                resp = await client.post(url, json={'prompt': prompt, 'max_tokens': max_tokens})
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get('usage', {})
                result = LLMResponse(text=data['text'], prompt_tokens=usage.get('prompt_tokens', 0), completion_tokens=usage.get('completion_tokens', 0), total_tokens=usage.get('total_tokens', 0))
                _health.record_success()
                logger.info('LLM call succeeded on attempt=%d | total_tokens=%d', attempt, result.total_tokens)
                return result
            if resp.status_code in (429, 500, 503):
                logger.warning('LLM returned status=%d on attempt=%d', resp.status_code, attempt)
                last_exception = LLMCallError(f'LLM returned HTTP {resp.status_code}')
                _health.record_failure()
                continue
            logger.error('LLM returned non-retryable status=%d on attempt=%d', resp.status_code, attempt)
            _health.record_failure()
            raise LLMCallError(f'LLM returned non-retryable HTTP {resp.status_code}')
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            logger.warning('LLM network error on attempt=%d: %s', attempt, exc)
            last_exception = exc
            _health.record_failure()
            continue
    raise LLMCallError(f'LLM exhausted {LLM_MAX_RETRIES} retries. Last error: {last_exception}')

def is_degraded(threshold_override: Optional[float]=None) -> bool:
    threshold = threshold_override if threshold_override is not None else DEGRADED_THRESHOLD_SECONDS
    return _health.is_degraded(threshold)

def reset_health() -> None:
    global _health
    _health = _HealthState()
