"""
Reusable fake/mock objects for testing the snap4events graph without any
real network call: no MCP server, no Tavily, no Snap4City ClearML endpoint.

Not a test file itself (no test_ functions) - imported by the other files
in this directory.
"""

import json
from types import SimpleNamespace


# ---------------------------------------------------------
# Fake MCP client: replaces the real fastmcp Client
# ---------------------------------------------------------

class FakeMCPClient:
    """Records every call_tool invocation (name + args) so tests can assert
    on exactly what was sent, e.g. regression-testing the 'city' vs
    'location' parameter bug."""

    def __init__(self, db_events=None, raise_on_get_events=False):
        self.db_events = db_events if db_events is not None else []
        self.raise_on_get_events = raise_on_get_events
        self.calls = []
        self.saved_payloads = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))

        if name == "get_events":
            if self.raise_on_get_events:
                raise RuntimeError("Simulated MCP failure")
            return SimpleNamespace(data=json.dumps(self.db_events))

        if name == "save_events_to_db":
            self.saved_payloads.append(args)
            return SimpleNamespace(data=json.dumps({"status": "ok"}))

        raise ValueError(f"Unexpected MCP tool called in test: {name}")


# ---------------------------------------------------------
# Fake LLM client: replaces ClearMLClient, no HTTP involved
# ---------------------------------------------------------

class FakeLLMClient:
    """A scriptable fake for the ClearML client. Pass a callable or a fixed
    string/exception as `response` to control what generate() returns."""

    def __init__(self, response=None, raise_error=None):
        self.response = response
        self.raise_error = raise_error
        self.prompts_received = []

    def generate(self, prompt=None, **kwargs):
        self.prompts_received.append(prompt)

        if self.raise_error is not None:
            raise self.raise_error

        if callable(self.response):
            return self.response(prompt)

        return self.response


# ---------------------------------------------------------
# Fakes for the web-search branch (Tavily + requests)
# ---------------------------------------------------------

class FakeTavilyClient:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def search(self, query, search_depth="basic", max_results=4):
        return {"results": [{"url": "https://example.com/evento-web"}]}


class FakeResponse:
    def __init__(self, content=b"<html><body>Test</body></html>", status_code=200, raise_error=None):
        self.content = content
        self.status_code = status_code
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error is not None:
            raise self._raise_error


def make_fake_requests_get(response=None, raise_error=None):
    """Returns a function with the same signature as requests.get, for
    monkeypatching nodes.requests.get."""

    def fake_get(url, timeout=10, headers=None):
        if raise_error is not None:
            raise raise_error
        return response if response is not None else FakeResponse()

    return fake_get
