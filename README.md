# cognocient

[![PyPI](https://img.shields.io/pypi/v/cognocient.svg)](https://pypi.org/project/cognocient/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A drop-in wrapper around the OpenAI and Anthropic Python SDKs that reports
usage to Cognocient asynchronously, so you get live cost attribution
without changing your `base_url` or routing traffic through a proxy.

```bash
pip install cognocient[openai]      # or cognocient[anthropic], or both
```

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After
from cognocient import CognocientOpenAI as OpenAI
client = OpenAI(
    api_key="sk-...",              # your own real OpenAI key, used exactly as before
    cognocient_key="sk-cog-...",   # the same proxy key you'd use with the Cognocient proxy
)

client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
    cognocient_feature="support-bot",   # optional attribution — same field names the proxy accepts as X-Cost-* headers
)
```

Every method the real SDK exposes still works unchanged. This wrapper only
intercepts `chat.completions.create()` (`messages.create()` for Anthropic)
to time the call and report its usage after the fact; everything else is
forwarded to the real client untouched.

## This is one of three ways to see your Cognocient dashboard

| | Live attribution | Pre-call enforcement (block/degrade) | Code change |
|---|---|---|---|
| **Proxy** (`base_url` swap) | Yes | Yes | One line |
| **This wrapper** | Yes | No — see below | Swap the import, add a key |
| **CSV/OTel import** | No (historical only) | No | None |

## Security — read this before you decide

**This wrapper is not more secure than the proxy. It is a different
tradeoff, not a strictly better one.**

With the proxy, your real provider API key lives server-side, under
Cognocient's control, in one place. With this wrapper, your real provider
key stays in your own application process, exactly as it does today
without Cognocient at all — the wrapper calls the provider directly,
using your key, inside your runtime. Some security teams prefer that
(no third-party network hop in the request path); others are less
comfortable with third-party code executing inside their process with
key access. Both are reasonable positions. We're not going to tell you
this "removes a security roadblock" — it trades one shape of exposure
for a different one.

What this wrapper honestly gives you over the proxy:
- **Zero added request latency.** Reporting happens after your real
  call already returned, on a background thread, off the critical path.
- **Zero risk of a Cognocient outage affecting your production call.**
  If Cognocient's ingestion API is down or unreachable, your call to
  OpenAI/Anthropic still completes normally — see "Reliability" below.

What you give up versus the proxy: pre-call enforcement. Because
Cognocient only hears about a call after it already happened, budgets
configured in Cognocient cannot block or degrade a call made through
this wrapper before it fires. The dashboard will say so explicitly for
any account using this path.

## Reliability

Reporting is fire-and-forget on a background thread with a bounded local
queue, flushed every few seconds or every 50 calls, whichever comes
first. If the ingestion API is slow, down, or unreachable:

- Your real provider call is completely unaffected — it already happened
  before reporting was attempted.
- No exception is ever raised into your code from a reporting failure.
- No retry loop that could pile up work in your process — a failed batch
  is dropped and logged locally at `DEBUG` level via the `cognocient`
  logger, not retried.

See `tests/test_reporter_failure_isolation.py` for a test that simulates
an unreachable ingestion endpoint and asserts the real call still
completes normally.

## Known limitation: streaming isn't reported yet

`stream=True` calls are passed through to the real SDK completely
unmodified — your application behaves identically — but are **not**
currently reported to Cognocient. Usage totals aren't available until a
stream completes, and reliably capturing them requires wrapping the
stream iterator itself, which this version doesn't do. If most of your
traffic streams, this wrapper will under-report your usage today. Use
the proxy or the CSV/OTel importer if that matters for your evaluation.

## Attribution fields

Same field names the proxy accepts as `X-Cost-*` headers, passed as
keyword arguments instead:

| Wrapper kwarg | Proxy header |
|---|---|
| `cognocient_feature` | `X-Cost-Feature` |
| `cognocient_department` | `X-Cost-Department` |
| `cognocient_user` | `X-Cost-User` |
| `cognocient_session` | `X-Cost-Session` |
| `cognocient_tier` | `X-Cost-Tier` |
| `cognocient_project` | `X-Cost-Project` |
| `cognocient_gl_account` | `X-Cost-GL-Account` |
| `cognocient_workload` | `X-Cost-Workload` |
| `cognocient_outcome` | `X-Cost-Outcome` |
| `cognocient_run_id` | `X-Cost-Run-ID` |

## Development

```bash
pip install -e ".[dev]"
pytest
python benchmark/benchmark_wrapper_overhead.py
```
