import asyncio
import json
import traceback
from pprint import pprint
from dotenv import load_dotenv
from pathlib import Path
from mcpuniverse.tracer.collectors import MemoryCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
import os

load_dotenv()

OUT_RESULTS = Path("benchmark_output.json")
OUT_TRACES = Path("benchmark_traces.json")

def print_trace_logs(trace_records):
    """
    Robust trace logger: handles dict entries and TraceRecord objects
    that expose .event, .data, .timestamp, or a .dict() method.
    """
    if not trace_records:
        print("No trace records available.")
        return

    def record_to_fields(e):
        # Return tuple (evt, data, ts) in a best-effort way.
        # e may be a dict or an object (TraceRecord).
        if isinstance(e, dict):
            evt = e.get("event")
            data = e.get("data") or e.get("payload") or {}
            ts = e.get("timestamp") or e.get("time")
            return evt, data, ts

        # object path: try common attributes
        evt = getattr(e, "event", None)
        data = getattr(e, "data", None)
        ts = getattr(e, "timestamp", None) or getattr(e, "time", None)

        # if .data is None, try .payload or .to_dict()
        if data is None:
            data = getattr(e, "payload", None)

        # Last resort: try .dict() or __dict__
        if data is None:
            try:
                d = e.dict() if hasattr(e, "dict") else None
            except Exception:
                d = None
            if d:
                evt = evt or d.get("event")
                data = d.get("data") or d.get("payload") or d.get("payloads") or {}
                ts = ts or d.get("timestamp") or d.get("time")
        if data is None:
            try:
                data = getattr(e, "__dict__", {})
            except Exception:
                data = {"repr": repr(e)}
        return evt, data or {}, ts

    print("\n==================== FULL TRACE LOG ====================")
    for e in trace_records:
        evt, data, ts = record_to_fields(e)
        prefix = f"[{ts}]" if ts is not None else ""
        # Normalize event name for comparison
        name = (evt or "").lower()

        # LLM entries
        if "llm_request" in name or name == "llm_request" or "request" in name and "llm" in name:
            print(f"\n{prefix} [LLM REQUEST]")
            prompt = data.get("prompt") or data.get("messages") or data.get("input") or data
            pprint(prompt)
        elif "llm_response" in name or name == "llm_response" or "response" in name and "llm" in name:
            print(f"\n{prefix} [LLM RESPONSE]")
            resp = data.get("response") or data.get("output") or data
            pprint(resp)
        # Tool interaction
        elif "tool_call" in name or name == "tool_call" or "tool" in name and "call" in name:
            print(f"\n{prefix} [TOOL CALL]")
            tool = data.get("tool") or data.get("name") or data.get("tool_name")
            print("tool:", tool)
            print("arguments:")
            # arguments may be nested differently
            args = data.get("arguments") or data.get("args") or data.get("params") or data
            pprint(args)
        elif "tool_result" in name or name == "tool_result" or "result" in name and "tool" in name:
            print(f"\n{prefix} [TOOL RESULT]")
            tool = data.get("tool") or data.get("name") or data.get("tool_name")
            print("tool:", tool)
            print("result:")
            res = data.get("result") or data.get("output") or data
            pprint(res)
        else:
            # Generic fallback — print event name and payload
            print(f"\n{prefix} [EVENT] {evt}")
            # If data is a mapping with a few keys, pretty print. Otherwise show repr.
            try:
                if isinstance(data, dict) and len(data) <= 8:
                    pprint(data)
                else:
                    # try to extract commonly useful keys
                    small = {}
                    for k in ("tool", "name", "arguments", "result", "prompt", "response", "messages"):
                        if isinstance(data, dict) and k in data:
                            small[k] = data[k]
                    if small:
                        pprint(small)
                    else:
                        # final fallback
                        print(repr(data))
            except Exception:
                print("Could not pretty-print event data; raw repr:")
                print(repr(data))

async def test():
    trace_collector = MemoryCollector()

    # Choose a benchmark config file under the folder "mcpuniverse/benchmark/configs"
    benchmark = BenchmarkRunner("dummy/benchmark_1.yaml")

    # Run the specified benchmark
    print("Running benchmark:", "dummy/benchmark_1.yaml")
    try:
        results = await benchmark.run(trace_collector=trace_collector)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Attempting to dump partial outputs.")
        results = getattr(benchmark, "_last_results", []) or []
    except Exception:
        print("\nException during benchmark.run():")
        traceback.print_exc()
        results = getattr(benchmark, "_last_results", []) or []

    print("Benchmark run completed.\n")

    # Print a minimal results summary (safe access)
    print("=== Results summary ===")
    for i, r in enumerate(results, start=1):
        # try common access patterns safely
        task_path = getattr(r, "task_path", None)
        if task_path is None:
            try:
                d = r.dict()
                task_path = d.get("task_path") or (d.get("task") or {}).get("path")
            except Exception:
                task_path = "<unknown>"
        success = getattr(r, "success", None)
        print(f"Task #{i}: path={task_path}  success={success}")

    # Get traces for the first task (safe)
    print("\n=== Traces for first task ===")
    if not results:
        print("No results available.")
        return

    first = results[0]

    # Safe extraction of task_trace_ids
    trace_ids = getattr(first, "task_trace_ids", None)
    if trace_ids is None:
        try:
            trace_ids = first.dict().get("task_trace_ids")
        except Exception:
            trace_ids = None

    # Attempt to pull the specific trace key used in the dummy benchmark
    trace_key = "dummy/tasks/weather_1.json"
    trace_records = None
    if trace_ids and trace_key in trace_ids:
        trace_id = trace_ids[trace_key]
        print("Trace ID for", trace_key, "=", trace_id)
        trace_records = trace_collector.get(trace_id)
        print("\n--- Trace records (raw) ---")
        pprint(trace_records)
    else:
        print(f"Trace key {trace_key!r} not found in first result's trace ids.")
        print("Available trace_ids:", trace_ids)
        # try to print any available trace for the first result
        if trace_ids:
            # pick first trace id available
            try:
                any_trace_id = next(iter(trace_ids.values()))
                print("Using first available trace id:", any_trace_id)
                trace_records = trace_collector.get(any_trace_id)
            except Exception:
                trace_records = None

    # Pretty-print per-event logs (LLM/tool)
    print_trace_logs(trace_records)

    # Also save serialized full results and trace data to disk
    print("\n=== Saving outputs to disk ===")
    # Serialize results with best-effort conversions
    serial_results = []
    for r in results:
        try:
            serial_results.append(r.dict())
        except Exception:
            try:
                serial_results.append(r.to_dict())
            except Exception:
                serial_results.append({"repr": repr(r)})

    try:
        OUT_RESULTS.write_text(json.dumps(serial_results, indent=2))
        print(f"Saved results -> {OUT_RESULTS}")
    except Exception:
        print("Failed to save results:")
        traceback.print_exc()

    # Save traces (map task_path -> events)
    trace_data = {}
    for r in results:
        # find trace ids
        trace_ids_local = None
        try:
            trace_ids_local = getattr(r, "task_trace_ids", None)
        except Exception:
            pass
        if trace_ids_local is None:
            try:
                trace_ids_local = (r.dict()).get("task_trace_ids", None)
            except Exception:
                trace_ids_local = None
        if not trace_ids_local:
            continue
        for task_path, trace_id in (trace_ids_local or {}).items():
            try:
                trace_data.setdefault(task_path, []).extend(trace_collector.get(trace_id) or [])
            except Exception:
                trace_data.setdefault(task_path, []).append({"error": "failed to read trace"})

    try:
        OUT_TRACES.write_text(json.dumps(trace_data, indent=2))
        print(f"Saved traces -> {OUT_TRACES}")
    except Exception:
        print("Failed to save traces:")
        traceback.print_exc()

    # Print full serialized results lightly (not too verbose)
    print("\n=== Full results (serialized) ===")
    pprint(serial_results)

if __name__ == "__main__":
    asyncio.run(test())