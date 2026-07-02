import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from proxy.config import CACHE_HIT_THRESHOLD, CACHE_TTL_SECONDS, DEGRADED_CACHE_THRESHOLD
from proxy.similarity import most_similar

logger = logging.getLogger('proxy.cache')

@dataclass
class CacheEntry:
    prompt: str
    response_text: str
    actual_tokens: int
    timestamp: float = field(default_factory=time.monotonic)
    is_degraded_origin: bool = False
_cache: dict[str, list[CacheEntry]] = {}

def _evict_expired(entries: list[CacheEntry]) -> list[CacheEntry]:
    now = time.monotonic()
    fresh = [e for e in entries if now - e.timestamp < CACHE_TTL_SECONDS]
    evicted = len(entries) - len(fresh)
    if evicted > 0:
        logger.debug('evicted %d expired cache entries', evicted)
    return fresh

def lookup(api_key: str, prompt: str, *, degraded_mode: bool=False) -> Optional[tuple[CacheEntry, float]]:
    if api_key not in _cache:
        return None
    _cache[api_key] = _evict_expired(_cache[api_key])
    entries = _cache[api_key]
    if not entries:
        return None
    threshold = DEGRADED_CACHE_THRESHOLD if degraded_mode else CACHE_HIT_THRESHOLD
    eligible = entries if degraded_mode else [e for e in entries if not e.is_degraded_origin]
    if not eligible:
        return None
    candidate_prompts = [e.prompt for e in eligible]
    best_score, best_idx = most_similar(prompt, candidate_prompts)
    if best_idx is not None and best_score >= threshold:
        matched = eligible[best_idx]
        logger.info('cache HIT | key=%s | score=%.4f | threshold=%.2f | degraded=%s', api_key, best_score, threshold, degraded_mode)
        return (matched, best_score)
    logger.debug('cache MISS | key=%s | best_score=%.4f | threshold=%.2f', api_key, best_score, threshold)
    return None

def store(api_key: str, prompt: str, response_text: str, actual_tokens: int, *, is_degraded_origin: bool=False) -> None:
    entry = CacheEntry(prompt=prompt, response_text=response_text, actual_tokens=actual_tokens, is_degraded_origin=is_degraded_origin)
    if api_key not in _cache:
        _cache[api_key] = []
    _cache[api_key] = _evict_expired(_cache[api_key])
    _cache[api_key].append(entry)
    logger.info('cache STORE | key=%s | tokens=%d | degraded=%s', api_key, actual_tokens, is_degraded_origin)

def get_cache_sizes() -> dict[str, int]:
    result: dict[str, int] = {}
    for key, entries in _cache.items():
        fresh = _evict_expired(entries)
        _cache[key] = fresh
        result[key] = len(fresh)
    return result
