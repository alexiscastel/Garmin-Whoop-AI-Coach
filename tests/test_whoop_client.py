from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from services.whoop.client import WhoopApiClient, WhoopConfiguration, WhoopConfigurationError


class _FakeWhoopClient:
    from_config_calls = []
    init_calls = []

    def __init__(self, token_info: object | None = object(), **kwargs):
        self.token_info = token_info
        self.saved_paths: list[str] = []
        type(self).init_calls.append({"token_info": token_info, **kwargs})

    @classmethod
    def reset(cls) -> None:
        cls.from_config_calls = []
        cls.init_calls = []

    @classmethod
    def from_config(cls, config_path: str, token_path: str):
        cls.from_config_calls.append((config_path, token_path))
        return cls(token_info=SimpleNamespace(access_token="reused-token", is_expired=False))

    def save_token(self, path: str) -> None:
        self.saved_paths.append(path)

    def close(self) -> None:
        return None


class _FakeOAuthHelper:
    init_calls = []
    auth_url_calls = 0
    opened = 0
    exchange_calls = []

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scopes: list[str]):
        type(self).init_calls.append(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "scopes": scopes,
            }
        )

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.auth_url_calls = 0
        cls.opened = 0
        cls.exchange_calls = []

    def get_authorization_url(self) -> str:
        type(self).auth_url_calls += 1
        return "http://example.test/oauth"

    def open_authorization_url(self) -> str:
        type(self).opened += 1
        return "http://example.test/oauth"

    async def exchange_code_for_token(self, session, code: str):
        type(self).exchange_calls.append(code)
        return {"access_token": "token"}


def test_whoop_client_reuses_existing_token(monkeypatch, tmp_path):
    _FakeWhoopClient.reset()
    token_path = tmp_path / "whoop.json"
    token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("services.whoop.client.load_whoopy_client_class", lambda: _FakeWhoopClient)

    client = WhoopApiClient(
        WhoopConfiguration(
            enabled=True,
            client_id="cid",
            client_secret="secret",
            token_path=str(token_path),
        )
    )

    whoop_client = client.connect()
    assert whoop_client is not None
    assert len(_FakeWhoopClient.from_config_calls) == 1
    assert len(_FakeWhoopClient.init_calls) == 1


def test_whoop_client_runs_browser_auth_when_no_token(monkeypatch, tmp_path):
    _FakeWhoopClient.reset()
    _FakeOAuthHelper.reset()
    token_path = tmp_path / "missing.json"
    monkeypatch.setattr("services.whoop.client.load_whoopy_client_class", lambda: _FakeWhoopClient)
    monkeypatch.setattr("services.whoop.client.load_whoopy_oauth_helper_class", lambda: _FakeOAuthHelper)
    monkeypatch.setattr("builtins.input", lambda _: "http://localhost:1234/?code=test-code")

    client = WhoopApiClient(
        WhoopConfiguration(
            enabled=True,
            client_id="cid",
            client_secret="secret",
            token_path=str(token_path),
        )
    )

    whoop_client = client.connect()
    assert whoop_client is not None
    assert len(_FakeWhoopClient.from_config_calls) == 0
    assert len(_FakeWhoopClient.init_calls) == 1
    assert _FakeOAuthHelper.init_calls[0]["redirect_uri"] == "http://localhost:1234"
    assert _FakeOAuthHelper.exchange_calls == ["test-code"]


def test_whoop_client_requires_credentials():
    client = WhoopApiClient(WhoopConfiguration(enabled=True))
    with pytest.raises(WhoopConfigurationError):
        client.connect()


def test_config_parser_uses_whoop_env_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("WHOOP_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "env-client-secret")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
athlete:
  name: "Tester"
  email: "user@example.com"

context:
  analysis: ""
  planning: ""

whoop:
  enabled: true
""",
        encoding="utf-8",
    )

    from cli.garmin_ai_coach_cli import ConfigParser

    config = ConfigParser(config_path).get_whoop_config()
    assert config.enabled is True
    assert config.client_id == "env-client-id"
    assert config.client_secret == "env-client-secret"


def test_whoop_client_uses_direct_recovery_collection_endpoint(monkeypatch, tmp_path):
    _FakeWhoopClient.reset()
    token_path = tmp_path / "whoop.json"
    token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("services.whoop.client.load_whoopy_client_class", lambda: _FakeWhoopClient)

    captured = {}

    def fake_get(url: str, headers: dict, params: dict, timeout: int):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "records": [{"cycle_id": 123, "score": {"recovery_score": 88}}],
            "next_token": None,
        }
        response.raise_for_status.return_value = None
        return response

    monkeypatch.setattr("services.whoop.client.requests.get", fake_get)

    client = WhoopApiClient(
        WhoopConfiguration(
            enabled=True,
            client_id="cid",
            client_secret="secret",
            token_path=str(token_path),
        )
    )

    records = client.get_recoveries("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert records == [{"cycle_id": 123, "score": {"recovery_score": 88}}]
    assert captured["url"] == "https://api.prod.whoop.com/developer/v2/recovery"
    assert captured["params"]["limit"] == 25
