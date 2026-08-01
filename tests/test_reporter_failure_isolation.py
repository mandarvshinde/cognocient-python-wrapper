"""
Proves the core reliability promise: if Cognocient's ingestion API is
unreachable, the customer's real provider call still completes normally
and no exception ever surfaces from the reporting path. Uses httpx.MockTransport
to fake the real OpenAI/Anthropic HTTP response (so this doesn't require
real API keys or network access to the providers), while pointing the
reporter at a real, guaranteed-unreachable address to force an actual
connection failure, not a mocked one.
"""

import logging

import httpx
import pytest

from cognocient import CognocientAnthropic, CognocientOpenAI

# Port 1 on loopback: nothing listens there, connection is refused immediately
# and reliably, in every environment (unlike a DNS-based "unreachable" host,
# which can vary by network config / CI sandboxing).
UNREACHABLE_INGEST_URL = "http://127.0.0.1:1/api/ingest/wrapper"

_OPENAI_CHAT_COMPLETION_JSON = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hello from the real provider"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_ANTHROPIC_MESSAGE_JSON = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "hello from the real provider"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 6},
}


def _mock_transport(json_body):
    def handler(request):
        return httpx.Response(200, json=json_body)
    return httpx.MockTransport(handler)


def test_openai_call_completes_and_raises_nothing_when_ingest_unreachable(caplog):
    mock_http_client = httpx.Client(transport=_mock_transport(_OPENAI_CHAT_COMPLETION_JSON))
    client = CognocientOpenAI(
        api_key="sk-test-fake",
        http_client=mock_http_client,
        cognocient_key="sk-cog-test",
        cognocient_ingest_url=UNREACHABLE_INGEST_URL,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        cognocient_feature="test-suite",
    )

    # The real call succeeded and its real content is unaffected by reporting.
    assert response.choices[0].message.content == "hello from the real provider"
    assert response.usage.prompt_tokens == 10

    # Force the reporter to actually attempt (and fail) delivery right now,
    # synchronously, rather than relying on the background timer — proves
    # the failure is caught, not merely "not yet attempted".
    with caplog.at_level(logging.DEBUG, logger="cognocient"):
        client._reporter.flush()  # must not raise
    assert any("failed to report" in r.message for r in caplog.records)


def test_anthropic_call_completes_and_raises_nothing_when_ingest_unreachable(caplog):
    mock_http_client = httpx.Client(transport=_mock_transport(_ANTHROPIC_MESSAGE_JSON))
    client = CognocientAnthropic(
        api_key="sk-ant-test-fake",
        http_client=mock_http_client,
        cognocient_key="sk-cog-test",
        cognocient_ingest_url=UNREACHABLE_INGEST_URL,
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        cognocient_feature="test-suite",
    )

    assert response.content[0].text == "hello from the real provider"
    assert response.usage.input_tokens == 12

    with caplog.at_level(logging.DEBUG, logger="cognocient"):
        client._reporter.flush()
    assert any("failed to report" in r.message for r in caplog.records)


def test_report_queue_full_drops_silently_without_raising():
    reporter_client = CognocientOpenAI(
        api_key="sk-test-fake",
        http_client=httpx.Client(transport=_mock_transport(_OPENAI_CHAT_COMPLETION_JSON)),
        cognocient_key="sk-cog-test",
        cognocient_ingest_url=UNREACHABLE_INGEST_URL,
    )
    reporter = reporter_client._reporter
    # Stop the background thread so the queue can't drain on its own, then
    # fill it past capacity to force queue.Full on the caller's own thread.
    reporter._stop = True
    reporter._wake.set()
    reporter._thread.join(timeout=2.0)

    from cognocient import CallReport
    for _ in range(reporter._queue.maxsize + 10):
        reporter.report(CallReport(
            model="gpt-4o-mini", provider="openai",
            prompt_tokens=1, completion_tokens=1, latency_ms=1,
        ))  # must never raise, even once the queue is full
