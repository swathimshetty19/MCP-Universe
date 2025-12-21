"""
YahooCheckpointBackend

Simple in-memory + file-persistent checkpoint backend for the Yahoo Finance MCP server.

Goal:
- Store "expensive" state (like full historical price data) on the server.
- Return a small checkpoint capsule (handle + digest + expiry) to the client.
- Later, tools can look up the state again using the handle instead of recomputing.
- Optionally persist state to a JSON file so it survives server restarts.

For real production, you'd probably swap this for Redis/Postgres/etc.,
but this is good enough for benchmarking and demos.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CheckpointRecord:
    """Internal record stored in the backend."""
    state: Dict[str, Any]
    digest: str
    expires_at: float  # unix timestamp
    summary: str


class YahooCheckpointBackend:
    """
    Minimal checkpoint backend for the Yahoo Finance server.

    Expected usage in tools:

        backend = ctx.session.checkpoint_backend  # may be None
        if backend is not None:
            cache_key = backend.make_cache_key(
                ticker, start_date, end_date, interval
            )
            cached_state = await backend.get_cached_state(cache_key)
            if cached_state is not None:
                # use cached_state["rows"]
                ...
            else:
                # fetch from yfinance, then:
                state = {
                    "type": "historical_prices",
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "rows": rows,
                    "cache_key": cache_key,
                }
                capsule = await backend.create_checkpoint(
                    session=ctx.session,
                    state=state,
                )

    The `capsule` is a small JSON-serializable dict that can safely be sent
    back to the client. The actual `state` stays in this backend (and on disk
    if storage_file is configured).
    """

    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        storage_file: str | Path | None = None,
    ) -> None:
        self._default_ttl = default_ttl_seconds
        self._store: Dict[str, CheckpointRecord] = {}
        # cache_key -> handle
        self._key_index: Dict[str, str] = {}

        # Optional JSON file for persistence
        self._storage_path: Path | None = (
            Path(storage_file).expanduser().resolve()
            if storage_file is not None
            else None
        )

        # Async lock to avoid races if multiple requests hit the same backend
        self._lock = asyncio.Lock()

        # Load existing data from disk if available
        if self._storage_path is not None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Helpers: key / digest and persistence
    # ------------------------------------------------------------------

    def make_cache_key(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> str:
        """
        Build a deterministic cache key for a given historical query.

        Servers can use this to decide whether to reuse a previous checkpoint
        instead of calling Yahoo Finance again.
        """
        return f"{ticker}|{start_date}|{end_date}|{interval}"

    def _compute_digest(self, state: Dict[str, Any]) -> str:
        """Compute a stable digest for the state dictionary.

        Uses default=str so that pandas.Timestamp / datetime / Decimal, etc.
        become JSON-serializable.
        """
        payload = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_from_disk(self) -> None:
        """Load store + index from JSON file, if it exists."""
        if self._storage_path is None:
            return
        if not self._storage_path.exists():
            return

        try:
            with self._storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            store_raw: Dict[str, Dict[str, Any]] = data.get("store", {}) or {}
            key_index: Dict[str, str] = data.get("key_index", {}) or {}

            self._store.clear()
            for handle, rec in store_raw.items():
                state = rec.get("state", {}) or {}
                digest = rec.get("digest", "") or ""
                expires_at = float(rec.get("expires_at", 0.0) or 0.0)
                summary = rec.get("summary", "") or ""
                self._store[handle] = CheckpointRecord(
                    state=state,
                    digest=digest,
                    expires_at=expires_at,
                    summary=summary
                )

            self._key_index = {str(k): str(v) for k, v in key_index.items()}
        except Exception:
            # If loading fails, we just start empty; no crash.
            self._store = {}
            self._key_index = {}

    def _save_to_disk_locked(self) -> None:
        """
        Save store + index to disk.

        Must be called only while holding self._lock.
        """
        if self._storage_path is None:
            return

        data = {
            "store": {
                handle: {
                    "state": rec.state,
                    "digest": rec.digest,
                    "expires_at": rec.expires_at,
                    "summary": rec.summary,
                }
                for handle, rec in self._store.items()
            },
            "key_index": self._key_index,
            "default_ttl": self._default_ttl,
        }

        tmp_path = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            # default=str so that any weird values in state get stringified
            json.dump(data, f, default=str)
        tmp_path.replace(self._storage_path)

    # ------------------------------------------------------------------
    # Public API used by the server / tools
    # ------------------------------------------------------------------

    async def get_cached_state(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Look up a previously stored state by cache_key.

        Returns the state dict if:
          - we have a handle for this cache_key
          - that handle exists
          - the record is not expired

        Otherwise returns None.

        This is what allows a *single* tool implementation to decide:
          - cache hit  -> use stored rows, no Yahoo call
          - cache miss -> call Yahoo, then create_checkpoint(state)
        """
        now = time.time()

        async with self._lock:
            handle = self._key_index.get(cache_key)
            if handle is None:
                return None

            record = self._store.get(handle)
            if record is None:
                # stale index entry
                self._key_index.pop(cache_key, None)
                self._save_to_disk_locked()
                return None

            if record.expires_at < now:
                # expired: clean up both maps
                self._store.pop(handle, None)
                self._key_index.pop(cache_key, None)
                self._save_to_disk_locked()
                return None

            # still valid
            return record.state

    async def create_checkpoint(
        self,
        *,
        session: Any,  # ServerSession, but we don’t depend on its type here
        state: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        summary: str | None = None,
    ) -> Dict[str, Any]:
        """
        Store `state` and return a lightweight capsule:

            {
                "handle": "...",
                "digest": "...",
                "expires_at": 1733350000.123,
                "ttl_seconds": 3600
            }

        The client can stash this capsule; the real state stays on the server.

        If `state` contains a `"cache_key"` field, we will additionally index
        that key to this handle so that future calls with the same arguments
        can reuse the stored state via get_cached_state().
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = time.time()
        expires_at = now + ttl
        digest = self._compute_digest(state)
        handle = f"yahoo_hist_{uuid.uuid4().hex}"

        record = CheckpointRecord(state=state, digest=digest, expires_at=expires_at, summary=summary)
        cache_key = state.get("cache_key")

        async with self._lock:
            self._store[handle] = record
            if cache_key:
                self._key_index[cache_key] = handle
            self._save_to_disk_locked()

        # This is what gets serialized into the tool result.
        return {
            "handle": handle,
            "digest": digest,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
            "summary": summary,
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
            # Optionally we could also delete+save here, but for validation
            # we just report expired and leave cleanup to resume/get_cached_state.
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
                # Clean up expired record
                self._store.pop(handle, None)
                # Also clean any cache_key pointing at this handle
                keys_to_remove = [k for k, h in self._key_index.items() if h == handle]
                for k in keys_to_remove:
                    self._key_index.pop(k, None)
                self._save_to_disk_locked()
                raise ValueError(f"Checkpoint expired: {handle}")

            # still valid – no mutation, no save needed
            return {
                "state": record.state,
                "digest": record.digest,
                "expires_at": record.expires_at,
            }

    async def delete_checkpoint(self, handle: str) -> None:
        """Delete a checkpoint handle, if it exists, and clean up index entries."""
        async with self._lock:
            record = self._store.pop(handle, None)
            if record is None:
                return

            # Remove any cache_key that pointed to this handle
            keys_to_remove = [k for k, h in self._key_index.items() if h == handle]
            for k in keys_to_remove:
                self._key_index.pop(k, None)

            self._save_to_disk_locked()