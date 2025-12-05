import asyncio
import json
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# How to start the Yahoo server as a subprocess
server_params = StdioServerParameters(
    command="python",  # or "python3" if needed on your machine
    args=["-m", "mcpuniverse.mcp.servers.yahoo_finance"],
)


def parse_history_result_WC(result, label: str):
    """
    Parse CallToolResult from get_historical_stock_prices.

    Returns a dict:
        {
          "rows": list,           # the price rows (or [])
          "checkpoint": dict|None,# checkpoint capsule (or None)
          "raw": dict|Any         # full parsed JSON (or whatever it was)
        }
    or None if we couldn't parse anything.
    """
    content = getattr(result, "content", None)
    if not content:
        print(f"{label}: no content.")
        return None

    # Find the first block that has .text
    block = next((b for b in content if getattr(b, "text", None) is not None), None)
    if block is None:
        print(f"{label}: no text content found in blocks.")
        return None

    text = getattr(block, "text", None)
    if text is None:
        print(f"{label}: selected block has no 'text'. Raw block:")
        print(repr(block))
        return None

    try:
        parsed = json.loads(text)
    except Exception as exc:
        print(f"{label}: could not parse text as JSON: {exc}")
        print(f"{label}: raw text (truncated): {text[:1000]}")
        return None

    # We expect the Yahoo tool to return a dict like:
    # {
    #   "ticker": ...,
    #   "start_date": ...,
    #   "end_date": ...,
    #   "interval": ...,
    #   "rows": [...],
    #   "checkpoint": {...}  # or null
    # }
    if not isinstance(parsed, dict):
        print(f"{label}: parsed JSON is not a dict, got {type(parsed)}")
        return {"rows": [], "checkpoint": None, "raw": parsed}

    rows = parsed.get("rows") or []
    checkpoint = parsed.get("checkpoint", None)

    return {
        "rows": rows,
        "checkpoint": checkpoint,
        "raw": parsed,
    }

def inspect_calltool_result(result, label="RESULT"):
    """
    Pretty-print everything inside a CallToolResult object.
    Helps you discover what fields exist, content structure,
    metadata, types, etc.
    """
    print("\n" + "="*80)
    print(f"🔍 {label}: FULL RESULT OBJECT")
    print("="*80)

    # 1. Show type
    print(f"type(result) = {type(result)}\n")

    # 2. Show all attributes on the object
    print("🧩 Attributes on result:")
    for attr in dir(result):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(result, attr)
        except Exception:
            val = "<error reading value>"
        print(f"  - {attr}: {val}")

    # 3. Inspect content blocks more deeply
    content = getattr(result, "content", None)
    print("\n📦 Content blocks:")
    if not content:
        print("  (no content)")
    else:
        for i, block in enumerate(content):
            print(f"  Block {i} type = {type(block)}")
            print(f"    repr: {block!r}")
            if hasattr(block, "text"):
                print(f"    text: {block.text[:300]}...")  # truncated
            else:
                print("    (no .text attribute)")

    print("="*80 + "\n")

def parse_history_result(result, label: str):
    """
    Given a CallToolResult from get_historical_stock_prices, try to:
      - extract the first content block's text
      - JSON-parse it
      - return the parsed object (or None on failure)
    """
    content = getattr(result, "content", None)
    if not content:
        print(f"{label}: no content.")
        return None

    block = content[0]
    text = getattr(block, "text", None)
    if text is None:
        print(f"{label}: first block has no 'text'. Raw block:")
        print(repr(block))
        return None

    try:
        parsed = json.loads(text)
        return parsed
    except Exception as exc:
        print(f"{label}: could not parse text as JSON: {exc}")
        print(f"{label}: raw text (truncated): {text[:1000]}")
        return None


async def phase_one_get_full_history(ticker: str):
    """
    Phase 1:
      - Start Yahoo server
      - Initialize MCP
      - Call get_historical_stock_prices over a long window
      - Print basic stats (row count, first 3 rows)
      - Disconnect (server exits)
    """
    print("\n=== PHASE 1: get_historical_stock_prices (full history) ===")

    t0 = time.perf_counter()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Phase 1: tools available:")
            for t in tools.tools:
                print(f"  - {t.name}")

            call_start = time.perf_counter()
            result = await session.call_tool(
                "get_historical_stock_prices",
                arguments={
                    "ticker": ticker,
                    "start_date": "2023-01-09",
                    "end_date": "2025-01-08",
                    "interval": "1d",
                },
            )
            call_end = time.perf_counter()
            print(f"\nPhase 1: call_tool latency: {call_end - call_start:.3f}s")

            #inspect_calltool_result(result, label="PHASE 1 result")
            parsed = parse_history_result_WC(result, "PHASE 1")
            '''if isinstance(parsed, list):
                print(f"PHASE 1: total rows = {len(parsed)}")
                print("PHASE 1: first 3 rows (truncated):")
                print(json.dumps(rows[:3], indent=2)[:2000])'''
            if isinstance(parsed, dict):
                rows = parsed.get("rows") or []
                checkpoint = parsed.get("checkpoint")
                raw = parsed["raw"]
                print(f"PHASE 1: total rows = {len(rows)}")
                print("PHASE 1: first 3 rows (truncated):")
                print(json.dumps(rows[:3], indent=2)[:2000])

                print("\nPHASE 1: checkpoint capsule:")
                print(json.dumps(checkpoint, indent=2)[:1000])

                debug_reason = raw.get("checkpoint_debug") if isinstance(raw, dict) else None
                print("\nPHASE 1: checkpoint_debug =", debug_reason)
            else:
                print("PHASE 1: parsed result is not a list/dict, raw parsed:")
                print(repr(parsed))

    t1 = time.perf_counter()
    print(f"\nPhase 1: total (spawn + init + call + shutdown): {t1 - t0:.3f}s")

    # Context you might reuse later (e.g., ticker)
    return {"ticker": ticker}


async def phase_two_last_five(ticker: str):
    """
    Phase 2:
      - Start a NEW Yahoo server (fresh process)
      - Initialize MCP again
      - Call the SAME tool: get_historical_stock_prices
      - This time, treat the "query" as: "I only care about the last 5 prices"
        by slicing the last 5 rows from the returned history.
    """
    print("\n=== PHASE 2: get_historical_stock_prices (last 5 prices only) ===")

    t0 = time.perf_counter()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Phase 2: tools available:")
            for t in tools.tools:
                print(f"  - {t.name}")

            # For now we call with the same date range; logically the "question"
            # is different (we only use last 5 rows). In a checkpoint world,
            # Phase 2 would reuse Phase 1's state instead of re-fetching.
            call_start = time.perf_counter()
            result = await session.call_tool(
                "get_historical_stock_prices",
                arguments={
                    "ticker": ticker,
                    "start_date": "2023-01-09",
                    "end_date": "2025-01-08",
                    "interval": "1d",
                },
            )
            call_end = time.perf_counter()
            print(f"\nPhase 2: call_tool latency: {call_end - call_start:.3f}s")

            #inspect_calltool_result(result, label="PHASE 2 result")
            parsed = parse_history_result_WC(result, "PHASE 2")
            '''if isinstance(parsed, list) and parsed:
                last_five = parsed[-5:]
                print(f"PHASE 2: total rows = {len(parsed)}")
                print("PHASE 2: last 5 rows (truncated):")
                print(json.dumps(last_five, indent=2)[:2000])'''
            if isinstance(parsed, dict):
                rows = parsed.get("rows") or []
                if rows:
                    last_five = rows[-5:]
                    print(f"PHASE 2: total rows = {len(rows)}")
                    print("PHASE 2: last 5 rows (truncated):")
                    print(json.dumps(last_five, indent=2)[:2000])
                else:
                    print("PHASE 2: no rows in result.")  
            else:
                print("PHASE 2: parsed result is not a non-empty list, raw parsed:")
                print(repr(parsed))

    t1 = time.perf_counter()
    print(f"\nPhase 2: total (spawn + init + call + shutdown): {t1 - t0:.3f}s")


async def main():
    ticker = "MSFT"

    # Phase 1: full-history style query
    context = await phase_one_get_full_history(ticker)

    # Phase 2: logically "give me last 5 prices" (same tool, same args,
    # but we only use the tail of the data)
    await phase_two_last_five(context["ticker"])


if __name__ == "__main__":
    asyncio.run(main())