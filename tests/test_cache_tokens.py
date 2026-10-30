"""Cache token counts reach the report, through the real SDK response objects."""
from _mock_http import mock_http_client

from cognocient import CognocientAnthropic, CognocientOpenAI
from cognocient._reporter import Reporter


def _client(cls, body, monkeypatch):
    reports = []
    monkeypatch.setattr(Reporter, "report", lambda self, r: reports.append(r))
    sdk = "anthropic" if cls is CognocientAnthropic else "openai"
    return cls(api_key="sk-test-fake", http_client=mock_http_client(sdk, body), cognocient_key="sk-cog-test"), reports


def test_anthropic_reports_all_input_with_cache_subsets_and_the_1h_split(monkeypatch):
    client, reports = _client(CognocientAnthropic, {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 3000,
                  "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
                  "output_tokens": 50},
    }, monkeypatch)
    client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    r = reports[0]
    assert (r.prompt_tokens, r.cache_read_tokens, r.cache_write_tokens, r.cache_write_1h_tokens, r.completion_tokens) == (4100, 1000, 3000, 2000, 50)


def test_anthropic_without_cache_fields(monkeypatch):
    client, reports = _client(CognocientAnthropic, {
        "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }, monkeypatch)
    client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    r = reports[0]
    assert (r.prompt_tokens, r.cache_read_tokens, r.cache_write_tokens, r.cache_write_1h_tokens) == (12, 0, 0, 0)


def test_openai_reports_cached_tokens(monkeypatch):
    client, reports = _client(CognocientOpenAI, {
        "id": "c1", "object": "chat.completion", "created": 1700000000, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 5, "total_tokens": 1205,
                  "prompt_tokens_details": {"cached_tokens": 1024}},
    }, monkeypatch)
    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    r = reports[0]
    assert (r.prompt_tokens, r.cache_read_tokens, r.cache_write_tokens) == (1200, 1024, 0)
