from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description='Customer identifier (acts as api_key; not real auth)')
    prompt: str = Field(..., min_length=1, description='The prompt text to send to the LLM')

class ChatResponse(BaseModel):
    text: str = Field(..., description='The completion text (from LLM or cache)')
    cached: bool = Field(..., description='True if the response was served from the semantic cache')
    degraded: bool = Field(..., description='True if the response is a degraded fallback (LLM unreachable)')
    estimated_tokens: int = Field(..., description='Heuristic token estimate for the prompt')
    actual_tokens: Optional[int] = Field(None, description='Actual total_tokens reported by the LLM.  Null when served from cache (no LLM call was made) or in degraded mode.')
    similarity_score: Optional[float] = Field(None, description='TF-IDF cosine similarity between this prompt and the matched cache entry.  Null on a cache miss that goes to the LLM normally.')

class MetricsResponse(BaseModel):
    cache_hit_rate: float = Field(..., description='cache hits (including degraded) / total requests')
    total_requests: int = Field(..., description='Total number of /v1/chat requests processed')
    tokens_saved: int = Field(..., description="Tokens saved by cache hits (sum of cached entries' actual token counts)")
    avg_estimation_error_pct: float = Field(..., description='Average |estimated - actual| / actual * 100 across all LLM-hitting requests.  0.0 if no LLM requests have been made yet.')
    degraded_responses: int = Field(..., description='Total number of degraded-mode responses served')
    budget_remaining: dict = Field(..., description='Per-api_key remaining token budget for the current window')

class RateLimitErrorDetail(BaseModel):
    error: str = 'budget_exceeded'
    message: str
    api_key: str
    budget_remaining: int
    budget_total: int
