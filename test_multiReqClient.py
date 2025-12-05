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


def pretty_print_call_result(result, label: str):
    """
    Helper to inspect a CallToolResult:
      - Print latency label
      - Inspect result.content (list of ContentBlock)
      - Try to JSON-parse any text block
    """
    print(f"\n=== {label} RAW RESULT OBJECT ===")
    print(repr(result))

    content = getattr(result, "content", None)
    if not content:
        print(f"{label}: no content field or empty content.")
        return

    print(f"\n{label}: content has {len(content)} block(s).")

    # Just look at the first block for now
    block = content[0]
    block_type = getattr(block, "type", None)
    print(f"{label}: first block type = {block_type}")

    # Most FastMCP servers use text blocks with JSON
    text = getattr(block, "text", None)
    if text is None:
        print(f"{label}: no 'text' on first block, raw block:")
        print(repr(block))
        return

    print(f"\n{label}: first block text (truncated):")
    print(text[:1000])

    # Try JSON parse
    try:
        parsed = json.loads(text)
        print(f"\n{label}: parsed JSON (truncated):")
        if isinstance(parsed, list):
            print(json.dumps(parsed[:3], indent=2)[:2000])
        elif isinstance(parsed, dict):
            subset = {k: parsed[k] for k in list(parsed.keys())[:10]}
            print(json.dumps(subset, indent=2)[:2000])
        else:
            print(json.dumps(parsed, indent=2)[:2000])
    except Exception as exc:
        print(f"\n{label}: text was not valid JSON: {exc}")


async def phase_one_get_history(ticker: str):
    """
    Phase 1:
      - Start Yahoo server
      - Initialize MCP
      - Call get_historical_stock_prices
      - Disconnect (server process exits)
    Returns:
      - dict with ticker (and maybe later some summary)
    """
    print("\n=== PHASE 1: get_historical_stock_prices (fresh server) ===")

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
            pretty_print_call_result(result, "PHASE 1")

    t1 = time.perf_counter()
    print(f"\nPhase 1: total (server spawn + init + call + shutdown): {t1 - t0:.3f}s")

    # After the context exits, the Yahoo server subprocess is gone.
    return {"ticker": ticker}


async def phase_two_related_request(context: dict):
    """
    Phase 2:
      - Start a *new* Yahoo server (fresh process)
      - Initialize MCP again
      - Make a related request using info from Phase 1 (same ticker)
    """
    ticker = context["ticker"]

    print("\n=== PHASE 2: get_stock_info (new server, related ticker) ===")

    t0 = time.perf_counter()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Phase 2: tools available:")
            for t in tools.tools:
                print(f"  - {t.name}")

            call_start = time.perf_counter()
            result = await session.call_tool(
                "get_stock_info",
                arguments={
                    "ticker": ticker,
                },
            )
            call_end = time.perf_counter()

            print(f"\nPhase 2: call_tool latency: {call_end - call_start:.3f}s")
            pretty_print_call_result(result, "PHASE 2")

    t1 = time.perf_counter()
    print(f"\nPhase 2: total (server spawn + init + call + shutdown): {t1 - t0:.3f}s")


async def main():
    ticker = "MSFT"

    # Phase 1: make a historical prices request, then disconnect
    context = await phase_one_get_history(ticker)

    # Phase 2: new process, new session, but semantically "related" query
    await phase_two_related_request(context)


if __name__ == "__main__":
    asyncio.run(main())