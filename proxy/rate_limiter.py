import asyncio
import logging
import time
from typing import Optional
from proxy.config import RATE_LIMIT_BUDGET, RATE_LIMIT_WINDOW_SECONDS
logger = logging.getLogger('proxy.rate_limiter')

class _KeyState:
    __slots__ = ('used_tokens', 'window_start', 'lock')

    def __init__(self) -> None:
        self.used_tokens: int = 0
        self.window_start: float = time.monotonic()
        self.lock: asyncio.Lock = asyncio.Lock()
_key_states: dict[str, _KeyState] = {}
_creation_lock: asyncio.Lock = asyncio.Lock()

async def _get_or_create_state(api_key: str) -> _KeyState:
    if api_key not in _key_states:
        async with _creation_lock:
            if api_key not in _key_states:
                _key_states[api_key] = _KeyState()
                logger.info('created budget state for key=%s', api_key)
    return _key_states[api_key]

def _maybe_reset_window(state: _KeyState) -> None:
    now = time.monotonic()
    if now >= state.window_start + RATE_LIMIT_WINDOW_SECONDS:
        state.used_tokens = 0
        state.window_start = now
        logger.debug('window reset')

async def check_and_reserve(api_key: str, estimated_tokens: int) -> tuple[bool, int]:
    state = await _get_or_create_state(api_key)
    async with state.lock:
        _maybe_reset_window(state)
        remaining_before = RATE_LIMIT_BUDGET - state.used_tokens
        if state.used_tokens + estimated_tokens > RATE_LIMIT_BUDGET:
            logger.info('key=%s | BUDGET EXCEEDED | used=%d estimate=%d limit=%d', api_key, state.used_tokens, estimated_tokens, RATE_LIMIT_BUDGET)
            return (False, remaining_before)
        state.used_tokens += estimated_tokens
        logger.debug('key=%s | reserved estimate=%d | used=%d/%d', api_key, estimated_tokens, state.used_tokens, RATE_LIMIT_BUDGET)
        return (True, RATE_LIMIT_BUDGET - state.used_tokens)

async def reconcile_budget(api_key: str, estimated_tokens: int, actual_tokens: int) -> None:
    state = await _get_or_create_state(api_key)
    async with state.lock:
        _maybe_reset_window(state)
        state.used_tokens = max(0, state.used_tokens - estimated_tokens + actual_tokens)
        logger.debug('key=%s | reconciled estimate=%d actual=%d | used=%d/%d', api_key, estimated_tokens, actual_tokens, state.used_tokens, RATE_LIMIT_BUDGET)

async def release_reservation(api_key: str, estimated_tokens: int) -> None:
    state = await _get_or_create_state(api_key)
    async with state.lock:
        state.used_tokens = max(0, state.used_tokens - estimated_tokens)
        logger.debug('key=%s | released estimate=%d | used=%d/%d', api_key, estimated_tokens, state.used_tokens, RATE_LIMIT_BUDGET)

def get_budget_remaining(api_key: str) -> int:
    state = _key_states.get(api_key)
    if state is None:
        return RATE_LIMIT_BUDGET
    _maybe_reset_window(state)
    return max(0, RATE_LIMIT_BUDGET - state.used_tokens)

def get_all_budgets() -> dict[str, int]:
    return {key: get_budget_remaining(key) for key in _key_states}
