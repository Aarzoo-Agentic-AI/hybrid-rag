# LLM layer

This package gives the rest of the project a single LangChain chat model per task (`answer`, `query_rewrite`). Behind that model, requests are retried, switched to a fallback provider when one is failing, and protected by circuit breakers. The calling code never imports a provider SDK directly.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.llm import LLMFactory

factory = LLMFactory(get_settings())  # create once per process
llm = factory.for_task("answer")  # a FailoverChatModel (a LangChain BaseChatModel)

message = llm.invoke("What is hybrid search?")
print(message.content)
print(message.response_metadata["failover"])  # which provider answered, attempts, latency
```

Because `FailoverChatModel` is a normal LangChain chat model, it works anywhere one does:

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({...})  # also .stream(), .ainvoke(), .astream()
```

## Files

| File | Contents |
|---|---|
| [`failover.py`](failover.py) | `FailoverChatModel` and `ProviderSlot` (one provider model plus its circuit breaker) |
| [`factory.py`](factory.py) | `LLMFactory`, which builds a `FailoverChatModel` for each task from `config.yaml`; `build_provider_model()` |
| [`errors.py`](errors.py) | `classify_error()` (retryable or fatal) and the LLM layer's exception types |
| [`circuit_breaker.py`](circuit_breaker.py) | `CircuitBreaker` and `CircuitBreakerRegistry` |

## How a request is handled

```
invoke(messages)
  │
  ├─ for each provider, in config order
  │    ├─ circuit open? ──► skip (recorded in "skipped")
  │    └─ for attempt 1..max_attempts
  │         ├─ overall deadline passed? ──► AllProvidersFailedError
  │         ├─ call the provider
  │         │    ├─ success ──► close the circuit, add metadata, return
  │         │    ├─ fatal error ──► raise it immediately (no failover)
  │         │    └─ retryable error ──► count a failure on the circuit
  │         │         ├─ circuit now open, or retries used up ──► next provider
  │         │         └─ otherwise ──► wait (backoff), retry
  │
  └─ no provider succeeded ──► AllProvidersFailedError (lists every failed attempt)
```

### Which errors are retried

`classify_error()` in [`errors.py`](errors.py) decides:

| Retryable: retry, then fail over | Fatal: raise immediately |
|---|---|
| Rate limits (429) | Bad request (400) |
| Server errors and overloaded responses (5xx, including Anthropic's 529) | Authentication and permission errors (401, 403) |
| 408, 409, 425 | Not found (404), for example a wrong model name |
| Timeouts and connection errors (OpenAI, Anthropic, `httpx`, built-in) | Any other exception, which is most likely a bug |

Fatal errors don't fail over on purpose: a bad request or an invalid key is a bug or a configuration problem, and switching providers would only hide it.

### Backoff

The wait before retry *n* is `min(max_delay_s, base_delay_s × 2^(n−1))`, randomised to between half and all of that value ("equal jitter"). A retry is skipped if the wait wouldn't fit within the overall deadline.

### Circuit breaker

Each provider model (for example `openai/gpt-5`) has its own breaker, and the tasks that use the same model share it.

- **Closed:** requests go through normally, and consecutive failures are counted.
- **Open:** after `failure_threshold` consecutive failures, the provider is skipped for `cooldown_s` seconds.
- **Half-open:** after the cooldown, one test request is let through. If it succeeds the circuit closes; if it fails, the circuit opens again. A test request that never reports back expires after another cooldown, so a provider can't stay stuck.

Only retryable errors count as failures. Fatal errors don't affect the breaker.

### Streaming

`stream()` and `astream()` can fail over **only before the first token arrives**. If a provider fails partway through an answer, `StreamInterruptedError` is raised instead of finishing the answer with a different model.

## Response metadata

Every response (and the merged result of a stream) contains `response_metadata["failover"]`:

```python
{
    "provider": "openai",
    "model": "gpt-5",
    "failed_over": False,  # True if a provider other than the first one answered
    "attempts": 1,  # total attempts, including failed ones
    "latency_s": 2.817,  # latency of the successful call
    "errors": [],  # one line for each failed attempt
    "skipped": [],  # providers skipped because their circuit was open
}
```

In a stream this information arrives in a separate empty chunk near the end, so combine the chunks (`full += chunk`) before reading it.

Token usage is in LangChain's standard `message.usage_metadata`.

## Configuration

All settings come from [`config.yaml`](../../../config.yaml) (see [`config.py`](../config.py)):

```yaml
llm:
  answer:                                   # ordered: primary first, then fallbacks
    - {provider: openai, model: gpt-5, reasoning_effort: low}
    - {provider: anthropic, model: claude-sonnet-5-5}
  query_rewrite:
    - {provider: openai, model: gpt-5-mini, reasoning_effort: minimal}
    - {provider: anthropic, model: claude-haiku-5-5}
  request_timeout_s: 60                     # timeout for each provider call
  overall_deadline_s: 120                   # total time across all attempts
  retry:
    max_attempts: 3                         # attempts for each provider
    base_delay_s: 0.5
    max_delay_s: 8
  circuit_breaker:
    failure_threshold: 5
    cooldown_s: 30
```

`reasoning_effort` (OpenAI reasoning models only) sets how long the model thinks before answering. Answering from sources that are already given needs little reasoning: with `low`, a `gpt-5` answer took about 6–8 s instead of about 26 s, and was just as correct.

- **API keys** come from `.env`: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. **A provider without a key is skipped.** If no provider for a task has a key, `NoProvidersConfiguredError` is raised.
- **The SDKs' own retries are turned off** (`max_retries=0`), so all retry decisions are made by `FailoverChatModel` and requests aren't retried twice.
- **Provider `fake`** returns a fixed reply and needs no key, which is useful for running the app without API keys:
  ```yaml
  answer:
    - {provider: fake, model: demo}
  ```

## Health

`LLMFactory.health()` returns the circuit-breaker state of every provider model built so far. The `/health` endpoint will use it:

```python
[{"name": "openai/gpt-5", "state": "closed", "consecutive_failures": 0}, ...]
```

Create **one** `LLMFactory` per process (for example at API startup), so every request shares the same circuit-breaker state.

## Tracing

Each provider attempt runs as a child of the failover run, named for example `openai/gpt-5 (attempt 2)`. With LangSmith switched on, a trace shows every attempt, including failed ones.

LangChain's `BaseChatModel.stream()` doesn't pass its run manager to `_stream()`, so without extra work the attempts in a streamed call would appear as separate, unlinked runs. `FailoverChatModel` therefore overrides `stream()` and `astream()`. Before calling LangChain's own implementation, it sets up the callback handlers and fixes the run ID, then gives that parent to `_stream()` through a context variable that's set only while each chunk is produced. `test_provider_attempts_are_nested_under_the_failover_run` checks the nesting for `invoke`, `stream`, `ainvoke` and `astream`.

## Exceptions

All of these inherit from `LLMError`, except fatal errors, which are raised as the original SDK exception:

| Exception | When |
|---|---|
| `AllProvidersFailedError` | Every provider failed or was skipped, or the deadline passed. `.failures` lists each attempt and `.skipped` lists skipped providers. |
| `StreamInterruptedError` | A stream failed after it had already sent tokens. |
| `NoProvidersConfiguredError` | None of a task's providers has an API key. |
| *(the original SDK error)* | Fatal errors, such as `openai.AuthenticationError`. |

## Adding a provider

1. Add the name to `ProviderName` in [`config.py`](../config.py).
2. Add a branch in `build_provider_model()` in [`factory.py`](factory.py) that returns the LangChain chat model, with the SDK's own retries turned off. Return `None` when its API key is missing.
3. If the SDK has its own exception types for timeouts or connection errors, add them to `_RETRYABLE_EXCEPTION_TYPES` in [`errors.py`](errors.py). HTTP status errors with a `status_code` attribute are already handled.
4. List it in `config.yaml`.

## Tests

| Command | What it runs |
|---|---|
| `uv run pytest` | Unit tests with scripted fake models; no API keys needed |
| `uv run pytest -m integration` | Live calls to the configured providers (skipped when no key is set) |
| `uv run python scripts/llm_smoke.py` | Sends one tiny prompt for each task and prints the provider, attempts, latency and tokens |

The unit tests are in [`tests/`](../../../tests/): `test_failover.py`, `test_circuit_breaker.py`, `test_llm_errors.py` and `test_llm_factory.py`. They raise the real OpenAI and Anthropic exception types and use a fake clock, so they run instantly and never sleep.
