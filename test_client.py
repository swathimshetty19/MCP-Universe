import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 1. How to start the Yahoo server as a subprocess
server_params = StdioServerParameters(
    command="python",  # or "python3" if your setup needs that
    args=["-m", "mcpuniverse.mcp.servers.yahoo_finance"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. Initialize MCP session
            await session.initialize()

            # 3. List tools
            tools = await session.list_tools()
            print("\n=== TOOLS EXPOSED BY YAHOO SERVER ===")
            for t in tools.tools:
                print(f"- {t.name} :: {t.description.splitlines()[0]}")

            # 4. Call get_historical_stock_prices as a test
            print("\n=== CALLING get_historical_stock_prices ===")
            result = await session.call_tool(
                "get_historical_stock_prices",
                arguments={
                    "ticker": "MSFT",
                    "start_date": "2023-01-09",
                    "end_date": "2025-01-08",
                    "interval": "1d",
                },
            )

            # result is a list[ContentBlock] (usually one block with JSON/text)
            print("\n=== RAW RESULT OBJECT ===")
            print(result)

            # If the server returns structured JSON in a text block, you can pretty-print:
            print("\n=== PRETTY RESULT (best-effort) ===")
            try:
                # Many FastMCP servers return a single text block with JSON inside
                block = result[0]
                if getattr(block, "type", None) == "text":
                    data = block.text
                else:
                    data = getattr(block, "text", str(block))

                # Try to parse as JSON, else just print string
                parsed = json.loads(data)
                print(json.dumps(parsed, indent=2)[:4000])  # trim if huge
            except Exception:
                # Fall back to just printing result repr
                print(result)


if __name__ == "__main__":
    asyncio.run(main())