import asyncio
import json
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================================================
# CONFIG
# =========================================================

SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "mcpuniverse.mcp.servers.yahoo_finance"],
)

TICKERS = [
    "MSFT", "AAPL", "GOOG", "AMZN", "META",
    "NVDA", "TSLA", "NFLX", "INTC", "AMD"
]

RESTARTS = 5

RESULTS_FILE = Path("exp2_results.jsonl")
HISTORY_DIR = Path("exp2_history_logs")
CAPSULE_FILE = Path("exp2_checkpoint_capsules.json")

HISTORY_DIR.mkdir(exist_ok=True)


# =========================================================
# UTILITIES
# =========================================================

def now_ms():
    return int(time.time() * 1000)


def make_args(ticker: str):
    return {
        "ticker": ticker,
        "start_date": "2023-01-09",
        "end_date": "2025-01-08",
        "interval": "1d",
    }


def parse_tool_result(result):
    content = getattr(result, "content", None)
    if not content:
        return None
    block = next((b for b in content if getattr(b, "text", None)), None)
    if not block:
        return None
    try:
        return json.loads(block.text)
    except Exception:
        return None


def estimate_context_size(obj):
    return len(json.dumps(obj, default=str).encode("utf-8"))


def log_result(record: dict):
    record["ts"] = now_ms()
    print(json.dumps(record, indent=2))
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def dump_history(agent: str, phase: str, payload):
    path = HISTORY_DIR / f"{agent}_{phase}.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


# =========================================================
# REPLAY HISTORY RESTORE (NEW)
# =========================================================

def load_replay_history():
    """
    Load full replay context history from disk (simulates cloud persistence).
    """
    path = HISTORY_DIR / "replay_cold.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


# =========================================================
# AGENTS
# =========================================================

class ReplayAgent:
    """
    Replay agent.
    On restart, reloads full prior context history from disk.
    """

    def __init__(self, *, restore: bool = False):  # <<< CHANGED >>>
        if restore:
            self.history = load_replay_history()
        else:
            self.history = []

    async def fetch(self, session, ticker, phase):
        args = make_args(ticker)

        t0 = time.perf_counter()
        result = await session.call_tool(
            "get_historical_stock_prices",
            arguments=args,
        )
        latency = (time.perf_counter() - t0) * 1000

        parsed = parse_tool_result(result) or {}
        rows = parsed.get("rows", [])
        source = parsed.get("source")

        self.history.append({
            "ticker": ticker,
            "rows_preview": rows,
            "rows": len(rows),
            "source": source,
        })

        dump_history("replay", phase, self.history)

        log_result({
            "agent": "replay",
            "phase": phase,
            "ticker": ticker,
            "latency_ms": round(latency, 2),
            "rows": len(rows),
            "source": source,
            "context_bytes": estimate_context_size(self.history),
            "used_handle": False,
        })


class CheckpointAgent:
    """
    Checkpoint agent with disk-persisted checkpoint capsules.
    """

    def __init__(self):
        self.handles = self._load_capsules()
        self.summaries = [v.get("summary", "") for v in self.handles.values()]

    def _load_capsules(self):
        if CAPSULE_FILE.exists():
            return json.loads(CAPSULE_FILE.read_text())
        return {}

    def _save_capsules(self):
        CAPSULE_FILE.write_text(json.dumps(self.handles, indent=2))

    async def fetch(self, session, ticker, phase):
        args = make_args(ticker)

        resume_latency = 0.0
        used_handle = False

        if ticker in self.handles:
            t0 = time.perf_counter()
            await session.call_tool(
                "checkpoint/resume",
                arguments={"handle": self.handles[ticker]["handle"]},
            )
            resume_latency = (time.perf_counter() - t0) * 1000
            used_handle = True

        t1 = time.perf_counter()
        result = await session.call_tool(
            "get_historical_stock_prices",
            arguments=args,
        )
        call_latency = (time.perf_counter() - t1) * 1000

        parsed = parse_tool_result(result) or {}
        rows = parsed.get("rows", [])
        source = parsed.get("source")
        checkpoint = parsed.get("checkpoint")

        if checkpoint:
            self.handles[ticker] = checkpoint
            self.summaries.append(checkpoint.get("summary", ""))
            self._save_capsules()

        dump_history("checkpoint", phase, {
            "handles": self.handles,
            "summaries": self.summaries,
        })

        log_result({
            "agent": "checkpoint",
            "phase": phase,
            "ticker": ticker,
            "latency_ms": round(call_latency if not used_handle else 0.0, 2),
            "resume_latency_ms": round(resume_latency, 2),
            "call_latency_ms": round(call_latency, 2),
            "rows": len(rows),
            "source": source,
            "stored_handle": checkpoint is not None,
            "used_handle": used_handle,
            "context_bytes": estimate_context_size(self.summaries),
        })


# =========================================================
# EXPERIMENT
# =========================================================

async def run():
    RESULTS_FILE.unlink(missing_ok=True)
    CAPSULE_FILE.unlink(missing_ok=True)

    print("\n=== EXPERIMENT: REPLAY (persisted) vs CHECKPOINT (resume) ===\n")

    # ---------------------------
    # COLD START: fetch ALL tickers
    # ---------------------------
    async with stdio_client(SERVER_PARAMS) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            replay = ReplayAgent()
            checkpoint = CheckpointAgent()

            for ticker in TICKERS:
                await replay.fetch(session, ticker, "cold")
                await checkpoint.fetch(session, ticker, "cold")

    # ---------------------------
    # RESTARTS: fetch ONLY MSFT
    # ---------------------------
    for i in range(1, RESTARTS + 1):
        print(f"\n=== RESTART {i} ===")

        async with stdio_client(SERVER_PARAMS) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()

                replay = ReplayAgent(restore=True)   # <<< CHANGED >>>
                checkpoint = CheckpointAgent()

                await replay.fetch(session, "MSFT", f"restart_{i}")
                await checkpoint.fetch(session, "MSFT", f"restart_{i}")


if __name__ == "__main__":
    asyncio.run(run())