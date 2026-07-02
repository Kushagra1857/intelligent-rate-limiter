import asyncio
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from proxy import cache, llm_client, metrics, rate_limiter
from proxy.config import MOCK_LLM_BASE_URL, RATE_LIMIT_BUDGET, RECOVERY_PROBE_INTERVAL_SECONDS
from proxy.llm_client import LLMCallError
from proxy.schemas import ChatRequest, ChatResponse, RateLimitErrorDetail
from proxy.token_heuristic import estimate_tokens

logger = logging.getLogger('proxy.main')

async def _recovery_probe_loop() -> None:
    while True:
        await asyncio.sleep(RECOVERY_PROBE_INTERVAL_SECONDS)
        if not llm_client.is_degraded():
            continue
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f'{MOCK_LLM_BASE_URL}/v1/health')
            if resp.status_code == 200:
                logger.info('recovery probe succeeded — resetting LLM health state (circuit re-closed)')
                llm_client.reset_health()
            else:
                logger.debug('recovery probe: LLM still unhealthy (status=%d)', resp.status_code)
        except Exception as exc:
            logger.debug('recovery probe: connection error — LLM still unreachable: %s', exc)

@asynccontextmanager
async def lifespan(app: FastAPI):
    probe_task = asyncio.create_task(_recovery_probe_loop())
    logger.info('recovery probe loop started (interval=%ds)', RECOVERY_PROBE_INTERVAL_SECONDS)
    try:
        yield
    finally:
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass
        logger.info('recovery probe loop stopped')

app = FastAPI(title='Proxy API — Intelligent Rate Limiter', version='1.0.0', lifespan=lifespan)

@app.post('/v1/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    api_key = request.api_key
    prompt = request.prompt
    estimated = estimate_tokens(prompt)
    logger.info('key=%s | estimated_tokens=%d | prompt_preview=%.40r', api_key, estimated, prompt)
    allowed, budget_after = await rate_limiter.check_and_reserve(api_key, estimated)
    if not allowed:
        remaining = rate_limiter.get_budget_remaining(api_key)
        error_body = RateLimitErrorDetail(message=f"Token budget exhausted for key '{api_key}'. Remaining: {remaining} tokens.", api_key=api_key, budget_remaining=remaining, budget_total=RATE_LIMIT_BUDGET)
        return JSONResponse(status_code=429, content=error_body.model_dump())
    cache_result = cache.lookup(api_key, prompt, degraded_mode=False)
    if cache_result is not None:
        entry, score = cache_result
        await rate_limiter.release_reservation(api_key, estimated)
        metrics.record_request()
        metrics.record_cache_hit(entry.actual_tokens)
        logger.info('key=%s | CACHE HIT | similarity=%.4f | degraded=False', api_key, score)
        return ChatResponse(text=entry.response_text, cached=True, degraded=False, estimated_tokens=estimated, actual_tokens=None, similarity_score=round(score, 4))
    if llm_client.is_degraded():
        logger.warning('key=%s | LLM in degraded mode (pre-call) | searching cache', api_key)
        await rate_limiter.release_reservation(api_key, estimated)
        return await _serve_degraded(api_key, prompt, estimated)
    llm_response = None
    try:
        llm_response = await llm_client.call_llm(prompt)
    except LLMCallError as exc:
        logger.error('key=%s | LLM call failed after retries: %s', api_key, exc)
        if llm_client.is_degraded():
            await rate_limiter.release_reservation(api_key, estimated)
            return await _serve_degraded(api_key, prompt, estimated)
        await rate_limiter.release_reservation(api_key, estimated)
        raise HTTPException(status_code=502, detail=f'Upstream LLM call failed: {exc}')
    actual_tokens = llm_response.total_tokens
    await rate_limiter.reconcile_budget(api_key, estimated, actual_tokens)
    cache.store(api_key, prompt, llm_response.text, actual_tokens, is_degraded_origin=False)
    metrics.record_request()
    metrics.record_llm_estimation(estimated, actual_tokens)
    logger.info('key=%s | LLM success | actual_tokens=%d', api_key, actual_tokens)
    return ChatResponse(text=llm_response.text, cached=False, degraded=False, estimated_tokens=estimated, actual_tokens=actual_tokens, similarity_score=None)

async def _serve_degraded(api_key: str, prompt: str, estimated: int) -> JSONResponse:
    degraded_result = cache.lookup(api_key, prompt, degraded_mode=True)
    if degraded_result is not None:
        entry, score = degraded_result
        metrics.record_request()
        metrics.record_cache_hit(entry.actual_tokens)
        metrics.record_degraded_response()
        logger.info('key=%s | DEGRADED response | similarity=%.4f | degraded_origin=%s', api_key, score, entry.is_degraded_origin)
        response = ChatResponse(text=entry.response_text, cached=True, degraded=True, estimated_tokens=estimated, actual_tokens=None, similarity_score=round(score, 4))
        return JSONResponse(status_code=200, content=response.model_dump())
    logger.warning('key=%s | DEGRADED mode but no suitable cache entry (threshold 0.50)', api_key)
    metrics.record_request()
    metrics.record_degraded_response()
    return JSONResponse(status_code=503, content={'error': 'service_degraded', 'message': 'The upstream LLM is unavailable and no suitable cached response exists for this prompt. Please try again later.', 'degraded': True})

@app.get('/v1/metrics')
async def get_metrics():
    budget_remaining = rate_limiter.get_all_budgets()
    return metrics.get_metrics(budget_remaining)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('proxy.main:app', host='0.0.0.0', port=8000, reload=False)
