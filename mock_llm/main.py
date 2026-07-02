import asyncio
import logging
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

INITIAL_MODE: str = os.getenv('MOCK_LLM_MODE', 'normal')
SLOW_DELAY_SECONDS: float = float(os.getenv('MOCK_LLM_SLOW_DELAY', '2.0'))
DOWN_SLEEP_SECONDS: float = float(os.getenv('MOCK_LLM_DOWN_SLEEP', '120.0'))
VALID_MODES = {'normal', 'slow', 'flaky', 'always_429', 'always_500', 'down', 'random'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [mock_llm] %(levelname)s %(message)s')
logger = logging.getLogger('mock_llm')
_state = {'mode': INITIAL_MODE, 'request_counter': 0}

app = FastAPI(title='Mock LLM Service', version='1.0.0')
_SPLIT_PATTERN = re.compile('[\\s\\.,;:!?(){}\\[\\]\\"\'`<>|/\\\\@#$%^&*+=~\\-]+')

def _count_tokens(text: str) -> int:
    pieces = _SPLIT_PATTERN.split(text)
    return sum((1 for p in pieces if p))

def _generate_completion(prompt: str) -> str:
    preview = prompt[:40].replace('\n', ' ')
    return f"Here is a response to your prompt: '{preview}...'. This is a mock completion."

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description='The input prompt text')
    max_tokens: int = Field(200, ge=1, le=4096, description='Maximum tokens to generate')

class ModeChangeRequest(BaseModel):
    mode: str = Field(..., description=f'One of: {sorted(VALID_MODES)}')

@app.post('/v1/completions')
async def completions(request: CompletionRequest):
    _state['request_counter'] += 1
    current_counter = _state['request_counter']
    mode = _state['mode']
    logger.info('request #%d | mode=%s | prompt_preview=%.40r', current_counter, mode, request.prompt)
    if mode == 'always_429':
        return JSONResponse(status_code=429, content={'error': 'rate_limited', 'message': 'Mock upstream rate limit hit.'})
    if mode == 'always_500':
        return JSONResponse(status_code=500, content={'error': 'internal_error', 'message': 'Mock upstream internal error.'})
    if mode == 'flaky':
        if current_counter % 2 != 0:
            logger.info('flaky mode: failing request #%d (odd)', current_counter)
            return JSONResponse(status_code=500, content={'error': 'flaky_error', 'message': f'Flaky failure on request #{current_counter}.'})
    if mode == 'down':
        logger.info('down mode: sleeping %.1fs to simulate unreachable service', DOWN_SLEEP_SECONDS)
        await asyncio.sleep(DOWN_SLEEP_SECONDS)
        return JSONResponse(status_code=503, content={'error': 'service_down', 'message': 'Service is down.'})
    if mode == 'slow':
        logger.info('slow mode: sleeping %.1fs', SLOW_DELAY_SECONDS)
        await asyncio.sleep(SLOW_DELAY_SECONDS)
    if mode == 'random':
        import random
        delay = random.uniform(0.2, 0.8)
        await asyncio.sleep(delay)
    completion_text = _generate_completion(request.prompt)
    prompt_tokens = _count_tokens(request.prompt)
    completion_tokens = _count_tokens(completion_text)
    total_tokens = prompt_tokens + completion_tokens
    return {'text': completion_text, 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens}}

@app.post('/v1/mode')
async def change_mode(request: ModeChangeRequest):
    if request.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{request.mode}'. Valid modes: {sorted(VALID_MODES)}")
    previous = _state['mode']
    _state['mode'] = request.mode
    _state['request_counter'] = 0
    logger.info('mode changed: %s → %s', previous, request.mode)
    return {'previous_mode': previous, 'current_mode': request.mode, 'request_counter_reset': True}

@app.get('/v1/health')
async def health():
    return {'status': 'ok', 'mode': _state['mode'], 'request_counter': _state['request_counter']}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('mock_llm.main:app', host='0.0.0.0', port=8001, reload=False)
