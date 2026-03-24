import asyncio
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

WHOOP_SCOPES = [
    "offline",
    "read:recovery",
    "read:cycles",
    "read:sleep",
    "read:profile",
    "read:body_measurement",
]


class WhoopIntegrationError(RuntimeError):
    pass


class WhoopConfigurationError(WhoopIntegrationError):
    pass


class WhoopAuthError(WhoopIntegrationError):
    pass


@dataclass(frozen=True)
class WhoopConfiguration:
    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:1234"
    token_path: str = "~/.whoop/garmin-ai-coach.json"

    @property
    def expanded_token_path(self) -> Path:
        return Path(os.path.expanduser(self.token_path))

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.client_id:
            raise WhoopConfigurationError("WHOOP is enabled but no client_id is configured")
        if not self.client_secret:
            raise WhoopConfigurationError("WHOOP is enabled but no client_secret is configured")


def load_whoopy_client_class() -> type[Any]:
    try:
        from whoopy import WhoopClient
    except ImportError as exc:  # pragma: no cover - depends on environment packaging
        raise WhoopConfigurationError(
            "WHOOP integration requires the 'whoopy' package. Install project dependencies first."
        ) from exc
    return WhoopClient


def load_whoopy_oauth_helper_class() -> type[Any]:
    try:
        from whoopy.utils.auth import OAuth2Helper
    except ImportError as exc:  # pragma: no cover - depends on environment packaging
        raise WhoopConfigurationError(
            "WHOOP integration requires the 'whoopy' package. Install project dependencies first."
        ) from exc
    return OAuth2Helper


def run_async_in_thread(awaitable: Any) -> Any:
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except Exception as exc:  # pragma: no cover - exercised via caller
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error := result.get("error"):
        raise error
    return result.get("value")


class WhoopApiClient:
    def __init__(self, config: WhoopConfiguration):
        self.config = config
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self.connect()
        if self._client is None:
            raise WhoopAuthError("WHOOP client is not connected")
        return self._client

    def connect(self) -> Any:
        if self._client is not None:
            return self._client

        self.config.validate()
        client_class = load_whoopy_client_class()
        token_path = self.config.expanded_token_path
        config_path = self._write_temp_config()

        try:
            token_exists = token_path.exists()
            if token_exists:
                logger.info("Attempting WHOOP token reuse from %s", token_path)
                client = client_class.from_config(
                    config_path=config_path,
                    token_path=str(token_path),
                )
                if getattr(client, "token_info", None) is None:
                    logger.info("WHOOP token file was unusable; falling back to fresh auth")
                    client = self._fresh_auth(client_class, token_path)
            else:
                logger.info("No WHOOP token file found; starting browser auth flow")
                client = self._fresh_auth(client_class, token_path)
        except WhoopIntegrationError:
            raise
        except Exception as exc:
            raise WhoopAuthError(f"WHOOP authentication failed: {exc}") from exc
        finally:
            try:
                os.unlink(config_path)
            except FileNotFoundError:
                pass

        self._client = client
        return client

    def _fresh_auth(self, client_class: type[Any], token_path: Path) -> Any:
        oauth_helper_class = load_whoopy_oauth_helper_class()
        oauth_helper = oauth_helper_class(
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            redirect_uri=self.config.redirect_uri,
            scopes=WHOOP_SCOPES,
        )

        auth_url = oauth_helper.get_authorization_url()
        print(f"Opening browser to: {auth_url}")
        oauth_helper.open_authorization_url()
        print(f"\nAfter authorization, you'll be redirected to: {self.config.redirect_uri}")
        redirect_url = input("Paste the full redirect URL here: ").strip()

        code = parse_qs(urlparse(redirect_url).query).get("code", [None])[0]
        if not code:
            raise WhoopAuthError("No authorization code found in redirect URL")

        async def exchange_code() -> Any:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                return await oauth_helper.exchange_code_for_token(session, code)

        token_info = run_async_in_thread(exchange_code())
        client = client_class(
            token_info=token_info,
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            redirect_uri=self.config.redirect_uri,
        )
        token_path.parent.mkdir(parents=True, exist_ok=True)
        client.save_token(str(token_path))
        logger.info("Saved WHOOP OAuth token to %s", token_path)
        return client

    def _write_temp_config(self) -> str:
        payload = {
            "whoop": {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "redirect_uri": self.config.redirect_uri,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="whoop_config_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle)
            return handle.name

    def get_profile(self) -> Any:
        return self.client.user.get_profile()

    def get_body_measurements(self) -> Any:
        return self.client.user.get_body_measurements()

    def get_cycles(self, start: Any, end: Any) -> list[Any]:
        return self.client.cycles.get_all(start=start, end=end, limit_per_page=25)

    def get_recoveries(self, start: Any, end: Any) -> list[Any]:
        return self._get_collection("recovery", start=start, end=end)

    def get_sleeps(self, start: Any, end: Any) -> list[Any]:
        return self.client.sleep.get_all(start=start, end=end, limit_per_page=25)

    def _get_collection(self, path: str, start: Any, end: Any) -> list[dict[str, Any]]:
        self._refresh_token_if_needed()
        token_info = getattr(self.client, "token_info", None)
        if token_info is None or not getattr(token_info, "access_token", None):
            raise WhoopAuthError("WHOOP access token is unavailable")

        headers = {
            "Authorization": f"Bearer {token_info.access_token}",
            "Accept": "application/json",
            "User-Agent": "garmin-ai-coach/1.0",
        }

        params: dict[str, Any] = {
            "limit": 25,
            "start": self._to_iso8601(start),
            "end": self._to_iso8601(end),
        }
        records: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            request_params = dict(params)
            if next_token:
                request_params["nextToken"] = next_token

            response = requests.get(
                f"https://api.prod.whoop.com/developer/v2/{path}",
                headers=headers,
                params=request_params,
                timeout=30,
            )

            if response.status_code == 404:
                raise WhoopIntegrationError(f"WHOOP endpoint not found for {path}")
            if response.status_code == 401:
                raise WhoopAuthError("WHOOP authorization failed while fetching data")
            response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                break

            page_records = payload.get("records", [])
            if isinstance(page_records, list):
                records.extend([record for record in page_records if isinstance(record, dict)])

            next_token = payload.get("next_token")
            if not next_token:
                break

        return records

    def _refresh_token_if_needed(self) -> None:
        token_info = getattr(self.client, "token_info", None)
        if token_info is None or not getattr(token_info, "is_expired", False):
            return

        refresh = getattr(self.client, "refresh_token", None)
        if callable(refresh):
            refresh()

    @staticmethod
    def _to_iso8601(value: Any) -> str:
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return str(value)

    def close(self) -> None:
        if self._client is None:
            return

        token_path = self.config.expanded_token_path
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.save_token(str(token_path))
        except Exception:
            logger.exception("Failed to persist WHOOP token to %s", token_path)

        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        finally:
            self._client = None
