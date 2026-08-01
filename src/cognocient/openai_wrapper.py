"""
Drop-in wrapper around openai.OpenAI that reports every completed
chat.completions.create() call to Cognocient asynchronously, without ever
blocking or delaying the real call to OpenAI, and without ever raising a
reporting failure back into the caller.

SECURITY NOTE (read before writing any customer-facing copy referencing
this file): this wrapper runs INSIDE your application process and holds
your real OpenAI API key to make the call directly. That is a DIFFERENT
security exposure than the Cognocient proxy (where the key lives
server-side, under Cognocient's control, in one place) — not a strictly
lesser one. This wrapper is NOT "more secure" than the proxy. Its honest,
defensible benefits are zero added request latency and zero risk of a
Cognocient outage affecting your production call. See ../README.md.

LIMITATION: usage reporting only covers non-streaming calls in this
version. Streaming calls (stream=True) are passed through to the real SDK
completely unmodified — your application behaves identically — but are
not currently reported to Cognocient, since usage totals aren't available
until a stream completes, and reliably capturing them requires wrapping
the stream iterator itself, which this version doesn't do. If most of
your traffic streams, this wrapper will under-report your usage today.
"""

import time
from typing import Any

from ._reporter import DEFAULT_INGEST_URL, CallReport, Reporter
from ._tags import pop_tags


class _ReportingCompletions:
    """Wraps client.chat.completions — every attribute except create()
    is forwarded untouched to the real SDK object."""

    def __init__(self, real_completions: Any, reporter: Reporter, provider: str):
        self._real = real_completions
        self._reporter = reporter
        self._provider = provider

    def create(self, *args, **kwargs):
        tags = pop_tags(kwargs)
        is_streaming = bool(kwargs.get("stream"))

        start = time.monotonic()
        # The real call always happens first, and its result/exception is
        # returned to the caller exactly as the real SDK would — nothing
        # about reporting can change this line's outcome.
        response = self._real.create(*args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        if is_streaming:
            return response  # not reported in this version — see module docstring

        try:
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None) or kwargs.get("model", "unknown")
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            self._reporter.report(CallReport(
                model=model,
                provider=self._provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status_code=200,
                **tags,
            ))
        except Exception:
            # A bug in this wrapper's own bookkeeping must never surface
            # to the caller — the real response above has already
            # returned successfully by this point.
            pass

        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class _ReportingChat:
    def __init__(self, real_chat: Any, reporter: Reporter, provider: str):
        self._real = real_chat
        self.completions = _ReportingCompletions(real_chat.completions, reporter, provider)

    def __getattr__(self, name):
        return getattr(self._real, name)


class CognocientOpenAI:
    """
    Drop-in replacement import for openai.OpenAI:

        from cognocient import CognocientOpenAI as OpenAI
        client = OpenAI(
            api_key="sk-...",          # your own real OpenAI key
            cognocient_key="sk-cog-...",  # your Cognocient proxy key, reused here
        )
        client.chat.completions.create(
            model="gpt-4o", messages=[...],
            cognocient_feature="support-bot",  # optional attribution, same field names as the proxy's X-Cost-* headers
        )

    Every method the real OpenAI client exposes still works unchanged —
    this class only intercepts chat.completions.create() to report usage
    after the fact; everything else is forwarded to the real client
    untouched via attribute delegation.
    """

    def __init__(self, *args, cognocient_key: str, cognocient_ingest_url: str = DEFAULT_INGEST_URL, **kwargs):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "cognocient.CognocientOpenAI requires the 'openai' package. "
                "Install it with: pip install cognocient[openai]"
            ) from e

        self._real = openai.OpenAI(*args, **kwargs)
        self._reporter = Reporter(cognocient_key, ingest_url=cognocient_ingest_url)
        self.chat = _ReportingChat(self._real.chat, self._reporter, "openai")

    def __getattr__(self, name):
        return getattr(self._real, name)
