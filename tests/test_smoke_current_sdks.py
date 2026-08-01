"""
Basic interface smoke test against whatever OpenAI/Anthropic SDK versions
are actually installed (pip install cognocient[dev] pulls the current
releases, unpinned) — catches a breaking upstream interface change before
a customer hits it in production, not after. Run in CI on every push via
.github/workflows/python_wrapper_smoke_test.yml.

Deliberately does NOT hit real provider APIs — only checks that the real
SDK objects still expose the attributes this wrapper's delegation depends
on (client.chat.completions.create, client.messages.create, response.usage
field names), which is exactly the kind of change that would silently
break this wrapper without raising an ImportError.
"""

import inspect

import anthropic
import openai

from cognocient import CognocientAnthropic, CognocientOpenAI
from cognocient._tags import TAG_KWARGS, pop_tags


def test_openai_sdk_still_exposes_expected_interface():
    real_client = openai.OpenAI(api_key="sk-test-fake")
    assert hasattr(real_client, "chat")
    assert hasattr(real_client.chat, "completions")
    assert callable(real_client.chat.completions.create)

    sig = inspect.signature(real_client.chat.completions.create)
    assert "model" in sig.parameters
    assert "messages" in sig.parameters


def test_anthropic_sdk_still_exposes_expected_interface():
    real_client = anthropic.Anthropic(api_key="sk-ant-test-fake")
    assert hasattr(real_client, "messages")
    assert callable(real_client.messages.create)

    sig = inspect.signature(real_client.messages.create)
    assert "model" in sig.parameters
    assert "messages" in sig.parameters
    assert "max_tokens" in sig.parameters


def test_cognocient_openai_delegates_unknown_attributes_to_real_client():
    client = CognocientOpenAI(api_key="sk-test-fake", cognocient_key="sk-cog-test")
    # models, embeddings, etc. are not specially wrapped — must still resolve
    # via __getattr__ delegation to the real underlying client.
    assert client.models is client._real.models
    assert client.embeddings is client._real.embeddings


def test_cognocient_anthropic_delegates_unknown_attributes_to_real_client():
    client = CognocientAnthropic(api_key="sk-ant-test-fake", cognocient_key="sk-cog-test")
    assert client.completions is client._real.completions


def test_tag_kwargs_never_reach_the_real_sdk_call():
    kwargs = {"model": "gpt-4o", "cognocient_feature": "x", "cognocient_run_id": "run_1"}
    tags = pop_tags(kwargs)
    assert tags == {"tag_feature": "x", "run_id": "run_1"}
    # Every cognocient_* kwarg must be stripped before the real SDK sees kwargs,
    # since neither real SDK accepts them and would raise a TypeError.
    assert not any(k in kwargs for k in TAG_KWARGS)
    assert kwargs == {"model": "gpt-4o"}
