import logging
from typing import Optional
logger = logging.getLogger('proxy.metrics')
_counters = {'total_requests': 0, 'cache_hits': 0, 'tokens_saved': 0, 'degraded_responses': 0}
_estimation_errors: list[float] = []

def record_request() -> None:
    _counters['total_requests'] += 1

def record_cache_hit(tokens_saved: int) -> None:
    _counters['cache_hits'] += 1
    _counters['tokens_saved'] += tokens_saved
    logger.debug('cache hit recorded | tokens_saved=%d | cumulative=%d', tokens_saved, _counters['tokens_saved'])

def record_llm_estimation(estimated_tokens: int, actual_tokens: int) -> None:
    if actual_tokens > 0:
        error_pct = abs(estimated_tokens - actual_tokens) / actual_tokens * 100
        _estimation_errors.append(error_pct)
        logger.debug('estimation | estimated=%d actual=%d error=%.1f%%', estimated_tokens, actual_tokens, error_pct)

def record_degraded_response() -> None:
    _counters['degraded_responses'] += 1

def get_metrics(budget_remaining: dict) -> dict:
    total = _counters['total_requests']
    hits = _counters['cache_hits']
    cache_hit_rate = hits / total if total > 0 else 0.0
    avg_error = sum(_estimation_errors) / len(_estimation_errors) if _estimation_errors else 0.0
    return {'cache_hit_rate': round(cache_hit_rate, 4), 'total_requests': total, 'tokens_saved': _counters['tokens_saved'], 'avg_estimation_error_pct': round(avg_error, 2), 'degraded_responses': _counters['degraded_responses'], 'budget_remaining': budget_remaining}

def reset_metrics() -> None:
    global _estimation_errors
    _counters['total_requests'] = 0
    _counters['cache_hits'] = 0
    _counters['tokens_saved'] = 0
    _counters['degraded_responses'] = 0
    _estimation_errors = []
