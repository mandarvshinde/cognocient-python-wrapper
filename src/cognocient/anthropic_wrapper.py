"""
Drop-in wrapper around anthropic.Anthropic that reports every completed
messages.create() call to Cognocient asynchronously, without ever
blocking or delaying the real call to Anthropic, and without ever raising
a reporting failure back into the caller.

See openai_wrapper.py's module docstring for the security framing (same
tradeoffs apply here — this is not "more secure" than the proxy) and the
streaming limitation (also identical: non-streaming only in this version).
"""

import time
from typing import Any

from ._reporter import DEFAULT_INGEST_URL, CallReport, Reporter
from ._tags import pop_tags


class _ReportingMessages:
    def __init__(self, real_messages: Any, reporter: Reporter, provider: str):
        self._real = real_messages
        self._reporter = reporter
        self._provider = provider

    def create(self, *args, **kwargs):
        tags = pop_tags(kwargs)
        is_streaming = bool(kwargs.get("stream"))

        start = time.monotonic()
        response = self._real.create(*args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        if is_streaming:
            return response  # not reported in this version — see module docstring

        try:
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None) or kwargs.get("model", "unknown")
            # Anthropic's usage field names differ from OpenAI's — input/output, not prompt/completion.
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
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
            pass

        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class CognocientAnthropic:
    """
    Drop-in replacement import for anthropic.Anthropic:

        from cognocient import CognocientAnthropic as Anthropic
        client = Anthropic(
            api_key="sk-ant-...",         # your own real Anthropic key
            cognocient_key="sk-cog-...",  # your Cognocient proxy key, reused here
        )
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024, messages=[...],
            cognocient_feature="support-bot",
        )
    """

    def __init__(self, *args, cognocient_key: str, cognocient_ingest_url: str = DEFAULT_INGEST_URL, **kwargs):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "cognocient.CognocientAnthropic requires the 'anthropic' package. "
                "Install it with: pip install cognocient[anthropic]"
            ) from e

        self._real = anthropic.Anthropic(*args, **kwargs)
        self._reporter = Reporter(cognocient_key, ingest_url=cognocient_ingest_url)
        self.messages = _ReportingMessages(self._real.messages, self._reporter, "anthropic")

    def __getattr__(self, name):
        return getattr(self._real, name)
