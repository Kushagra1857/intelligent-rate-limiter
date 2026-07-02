# Intelligent Rate Limiter — LLM Proxy

This is a FastAPI-based proxy I built that sits in front of a mock LLM service. It does three main things — limits token usage per API key, caches semantically similar responses so you don't burn tokens on repeated questions, and handles upstream failures gracefully without crashing the whole system.

---

## Project Structure

```
intelligent-rate-limiter/
├── mock_llm/main.py        # fake upstream LLM
├── proxy/
│   ├── main.py             # routes only
│   ├── config.py           # all config constants
│   ├── schemas.py          # request/response models
│   ├── token_heuristic.py  # estimate token count from text
│   ├── similarity.py       # TF-IDF cosine similarity
│   ├── rate_limiter.py     # per-key hourly budget
│   ├── cache.py            # semantic cache with TTL
│   ├── llm_client.py       # retry logic + degraded mode
│   └── metrics.py          # request counters
└── tests/
    ├── test_token_heuristic.py
    ├── test_similarity.py
    └── test_scenarios.py   # 6 end-to-end scenarios
```

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running

You need two terminals open, both in the project root with the venv active.

**Terminal 1 — Mock LLM (port 8001):**

```bash
uvicorn mock_llm.main:app --host 0.0.0.0 --port 8001
```

**Terminal 2 — Proxy (port 8000):**

```bash
uvicorn proxy.main:app --host 0.0.0.0 --port 8000
```

Once both are up, all requests go through the proxy on port 8000. The proxy internally calls the mock LLM on port 8001.

---

## Tests

```bash
pytest tests/ -v
```

Add `-s` to also print the token validation table and similarity scores in the output.

---

## Testing

**Use Postman because** curl on Windows PowerShell has annoying quoting issues with JSON that'll waste your time. In Postman, you just:

1. Set method to `POST`, URL to `http://localhost:8000/v1/chat`
2. Go to Body → raw → JSON
3. Paste your json input and click Send

---

## Quick Examples (PowerShell)

**Send a prompt:**

```powershell
$body = @{
    api_key = "varahe_user_1"
    prompt = "What are the famous food streets near aminabad in Lucknow?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/v1/chat" -Method Post -ContentType "application/json" -Body $body
```

Response:

```json
{
  "text": "depends on the mock llm response",
  "cached": "boolean value depending on the entered prompt and cache",
  "degraded": "boolean value",
  "estimated_tokens": "depends on the prompt",
  "actual_tokens": "depends on the response of the prompt",
  "similarity_score": "calculated on the basis of the input prompt"
}
```

Send the same prompt again — you'll get `"cached": true` and `"actual_tokens": null`.

**Check metrics:**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/metrics"
```

---

## Mock LLM Modes

Change the mock's behavior at runtime without restarting it:

```powershell
$modeBody = @{ mode = "down" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/v1/mode" -Method Post -ContentType "application/json" -Body $modeBody
```

| LLM Mode     | Effect                                    |
| ------------ | ----------------------------------------- |
| `normal`     | Responds normally                         |
| `slow`       | Adds a 2s delay                           |
| `flaky`      | Fails on every odd request                |
| `always_429` | Always returns 429                        |
| `always_500` | Always returns 500                        |
| `down`       | Sleeps 120s (simulates unreachable)       |
| `random`     | Added 200-800ms delay (not used in tests) |

Check current mode:

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/v1/health"
```

---

## Environment Variables

All defaults work out of the box. Override as needed:

| Variable                     | Default                 | Notes                                          |
| ---------------------------- | ----------------------- | ---------------------------------------------- |
| `MOCK_LLM_BASE_URL`          | `http://127.0.0.1:8001` | Where the proxy looks for the LLM              |
| `LLM_REQUEST_TIMEOUT`        | `5.0`                   | Seconds before a request times out             |
| `LLM_MAX_RETRIES`            | `3`                     | Retry attempts on failure                      |
| `LLM_BACKOFF_BASE`           | `0.5`                   | Base delay for exponential backoff             |
| `LLM_BACKOFF_JITTER`         | `0.2`                   | Random jitter added to backoff                 |
| `DEGRADED_THRESHOLD_SECONDS` | `30.0`                  | How long to wait before entering degraded mode |
| `RATE_LIMIT_BUDGET`          | `10000`                 | Tokens per key per hour                        |
| `RATE_LIMIT_WINDOW_SECONDS`  | `3600`                  | Window size in seconds                         |
| `CACHE_TTL_SECONDS`          | `300`                   | Cache entry lifetime (5 min)                   |
| `CACHE_HIT_THRESHOLD`        | `0.80`                  | Similarity score needed for a cache hit        |
| `DEGRADED_CACHE_THRESHOLD`   | `0.50`                  | Lower bar for degraded-mode fallback           |
| `MOCK_LLM_SLOW_DELAY`        | `2.0`                   | Delay for `slow` mode                          |
| `MOCK_LLM_DOWN_SLEEP`        | `120.0`                 | Sleep duration for `down` mode                 |

---

## Known Limitations

- **No persistence** — everything is in-memory. Restart means data is lost.
- **Fixed rate limit window** — not a sliding window, so bursts at window boundaries can double-spend.
- **Lazy cache cleanup** — expired entries aren't removed until the next lookup for that key.
- **No real auth** — the `api_key` field is just a namespace, not validated.
- **Regex tokenizer** — works well against the mock LLM, but would have large errors against real BPE tokenizers.
- **Linear cache scan** — fine for this scale, but would need an ANN index in production.
- **Single process only** — the asyncio locks don't help across multiple replicas.
