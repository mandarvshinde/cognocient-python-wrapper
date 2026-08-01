"""
Real, measured latency benchmark for the cognocient Python wrapper's
call-path overhead.

Methodology: compares (a) a direct call through the real openai.OpenAI
client against (b) the identical call through CognocientOpenAI, both
hitting the SAME in-process httpx.MockTransport (so provider network
variance is excluded from both legs and cannot advantage either side).
This isolates exactly what the wrapper's own interception code — timing,
tag stripping, and queuing the report — costs on the customer's request
path. It does NOT include reporting delivery time, because reporting
runs on a background thread the request path never waits on; that's the
architectural claim this benchmark exists to check, not assume.

What's faked and why (disclosed here, not hidden):
  - No real network call to OpenAI or to Cognocient's ingestion API —
    both would introduce variance neither leg of this comparison is
    trying to measure. Real network latency to either service is NOT
    part of these numbers; only the wrapper's own added CPU work on the
    request path is.
  - The reporting queue is left completely unconsumed during the timed
    loop (the background thread's flush interval is longer than the
    whole benchmark run), so a measured cost that includes
    queue.put_nowait() reflects the actual per-call cost, not a case
    where a concurrently-draining queue happened to be cheap.

Run: python benchmark/benchmark_wrapper_overhead.py
"""

import statistics
import time

import httpx
import openai

from cognocient import CognocientOpenAI

ITERATIONS = 2000
WARMUP = 200

_CHAT_COMPLETION_JSON = {
    "id": "chatcmpl-bench",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


def _mock_transport():
    def handler(request):
        return httpx.Response(200, json=_CHAT_COMPLETION_JSON)
    return httpx.MockTransport(handler)


def _time_calls(create_fn, iterations):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        create_fn()
        samples.append((time.perf_counter() - start) * 1000)  # ms
    return samples


def _pct(samples, p):
    return statistics.quantiles(samples, n=100)[p - 1]


def main():
    real_client = openai.OpenAI(api_key="sk-bench-fake", http_client=httpx.Client(transport=_mock_transport()))
    wrapped_client = CognocientOpenAI(
        api_key="sk-bench-fake",
        http_client=httpx.Client(transport=_mock_transport()),
        cognocient_key="sk-cog-bench",
        # Deliberately unreachable (nothing listens on loopback port 1) —
        # if a future regression accidentally made the request path wait
        # on delivery, that would show up here as a multi-second outlier,
        # not a subtle few-ms difference easy to miss.
        cognocient_ingest_url="http://127.0.0.1:1/api/ingest/wrapper",
    )

    def raw_call():
        real_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    def wrapped_call():
        wrapped_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            cognocient_feature="benchmark",
        )

    _time_calls(raw_call, WARMUP)
    _time_calls(wrapped_call, WARMUP)

    raw_samples = _time_calls(raw_call, ITERATIONS)
    wrapped_samples = _time_calls(wrapped_call, ITERATIONS)

    print(f"Iterations per leg: {ITERATIONS}\n")
    print(f"{'':20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'mean (ms)':>10}")
    print(f"{'Raw SDK call':20} {_pct(raw_samples, 50):>10.4f} {_pct(raw_samples, 95):>10.4f} "
          f"{_pct(raw_samples, 99):>10.4f} {statistics.mean(raw_samples):>10.4f}")
    print(f"{'Wrapped call':20} {_pct(wrapped_samples, 50):>10.4f} {_pct(wrapped_samples, 95):>10.4f} "
          f"{_pct(wrapped_samples, 99):>10.4f} {statistics.mean(wrapped_samples):>10.4f}")

    added_p50 = _pct(wrapped_samples, 50) - _pct(raw_samples, 50)
    added_mean = statistics.mean(wrapped_samples) - statistics.mean(raw_samples)
    print(f"\nAdded overhead vs raw SDK — p50: {added_p50:.4f} ms, mean: {added_mean:.4f} ms")
    print("\n(Real network latency to OpenAI/Anthropic and to Cognocient's ingestion API")
    print(" is NOT included above — see this script's module docstring for why.)")


if __name__ == "__main__":
    main()
