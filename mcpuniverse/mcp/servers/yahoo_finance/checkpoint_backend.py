"""
YahooCheckpointBackend

Simple in-memory checkpoint backend for the Yahoo Finance MCP server.

Goal:
- Store "expensive" state (like full historical price data) on the server.
- Return a small checkpoint capsule (handle + digest + expiry) to the client.
- Later, tools can look up the state again using the handle instead of recomputing.

This is intentionally minimal and in-memory. For a real deployment, you’d swap
the internal dict for Redis/Postgres/etc.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CheckpointRecord:
    """Internal record stored in the backend."""
    state: Dict[str, Any]
    digest: str
    expires_at: float  # unix timestamp


class YahooCheckpointBackend:
    """
    Minimal checkpoint backend for the Yahoo Finance server.

    Expected usage in tools:

        backend = ctx.session.checkpoint_backend  # may be None
        if backend is not None:
            capsule = await backend.create_checkpoint(
                session=ctx.session,
                state={
                    "type": "historical_prices",
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "rows": history_rows,
                },
            )
            # include `capsule` in your tool output

    The `capsule` is a small JSON-serializable dict that can safely be sent
    back to the client. The actual `state` stays in this backend.
    """

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: Dict[str, CheckpointRecord] = {}
        # Async lock to avoid races if multiple requests hit the same backend
        self._lock = asyncio.Lock()

    def _compute_digest(self, state: Dict[str, Any]) -> str:
        """Compute a stable digest for the state dictionary."""
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def create_checkpoint(
        self,
        *,
        session: Any,  # ServerSession, but we don’t depend on its type here
        state: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Store `state` and return a lightweight capsule:

            {
                "handle": "...",
                "digest": "...",
                "expires_at": 1733350000.123
            }

        The client can stash this capsule; the real state stays on the server.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = time.time()
        expires_at = now + ttl
        digest = self._compute_digest(state)
        handle = f"yahoo_hist_{uuid.uuid4().hex}"

        record = CheckpointRecord(state=state, digest=digest, expires_at=expires_at)

        async with self._lock:
            self._store[handle] = record

        # This is what gets serialized into the tool result.
        return {
            "handle": handle,
            "digest": digest,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
        }

    async def validate_checkpoint(
        self,
        handle: str,
        expected_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a checkpoint handle.

        Returns a dict like:

            {
                "valid": True/False,
                "reason": str | None,
                "digest_matches": True/False | None,
                "expires_at": float | None,
            }
        """
        now = time.time()

        async with self._lock:
            record = self._store.get(handle)

        if record is None:
            return {
                "valid": False,
                "reason": "HANDLE_NOT_FOUND",
                "digest_matches": None,
                "expires_at": None,
            }

        if record.expires_at < now:
            return {
                "valid": False,
                "reason": "EXPIRED",
                "digest_matches": None,
                "expires_at": record.expires_at,
            }

        digest_matches: Optional[bool] = None
        if expected_digest is not None:
            digest_matches = record.digest == expected_digest
            if not digest_matches:
                return {
                    "valid": False,
                    "reason": "STALE_CONTENT",
                    "digest_matches": False,
                    "expires_at": record.expires_at,
                }

        return {
            "valid": True,
            "reason": None,
            "digest_matches": digest_matches,
            "expires_at": record.expires_at,
        }

    async def resume_checkpoint(self, handle: str) -> Dict[str, Any]:
        """
        Resume from a checkpoint. Raises KeyError or ValueError if invalid/expired.

        Returns:
            {
                "state": { ... },
                "digest": "...",
                "expires_at": float,
            }
        """
        now = time.time()

        async with self._lock:
            record = self._store.get(handle)

        if record is None:
            raise KeyError(f"Checkpoint handle not found: {handle}")

        if record.expires_at < now:
            raise ValueError(f"Checkpoint expired: {handle}")

        return {
            "state": record.state,
            "digest": record.digest,
            "expires_at": record.expires_at,
        }

    async def delete_checkpoint(self, handle: str) -> None:
        """Delete a checkpoint handle, if it exists."""
        async with self._lock:
            self._store.pop(handle, None)