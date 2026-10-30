"""Mock HTTP clients for the provider SDKs.

anthropic 2.x moved from httpx to httpx2 and rejects an httpx.Client passed
as http_client, so the mock comes from whichever library the SDK uses.
"""
import importlib


def _http_lib(sdk: str):
    version = getattr(importlib.import_module(sdk), "__version__", "0")
    if sdk == "anthropic" and int(version.split(".")[0]) >= 2:
        return importlib.import_module("httpx2")
    return importlib.import_module("httpx")


def mock_http_client(sdk: str, body: dict):
    """An http_client for `sdk` ("openai" or "anthropic") that answers every request with `body`."""
    http = _http_lib(sdk)
    return http.Client(transport=http.MockTransport(lambda request: http.Response(200, json=body)))
