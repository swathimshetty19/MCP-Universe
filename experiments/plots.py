import json
import matplotlib.pyplot as plt

LOG_FILE = "exp2_results.jsonl"


# -----------------------------
# Helpers
# -----------------------------
def save_and_show(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()


# -----------------------------
# 1. Cold start latency
# -----------------------------
replay_cold_lat = []
checkpoint_cold_lat = []

with open(LOG_FILE, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"] == "cold":
            if r["agent"] == "replay":
                replay_cold_lat.append(r["latency_ms"])
            elif r["agent"] == "checkpoint":
                checkpoint_cold_lat.append(r["call_latency_ms"])

x_replay = range(1, len(replay_cold_lat) + 1)
x_checkpoint = range(1, len(checkpoint_cold_lat) + 1)

plt.figure()
plt.plot(x_replay, replay_cold_lat, marker="o", label="Replay (cold)")
plt.plot(x_checkpoint, checkpoint_cold_lat, marker="o", label="Checkpoint (cold)")
plt.xlabel("Query index")
plt.ylabel("Latency (ms)")
plt.title("Cold Start Latency")
plt.legend()
save_and_show("exp2_cold_start_latency.png")


# -----------------------------
# 2. Cold start context bytes
# -----------------------------
replay_cold_ctx = []
checkpoint_cold_ctx = []

with open(LOG_FILE, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"] == "cold":
            if r["agent"] == "replay":
                replay_cold_ctx.append(r["context_bytes"])
            elif r["agent"] == "checkpoint":
                checkpoint_cold_ctx.append(r["context_bytes"])

plt.figure()
plt.plot(x_replay, replay_cold_ctx, marker="o", label="Replay context bytes")
plt.plot(x_checkpoint, checkpoint_cold_ctx, marker="o", label="Checkpoint context bytes")
plt.xlabel("Query index")
plt.ylabel("Context size (bytes)")
plt.title("Cold Start Context Growth")
plt.legend()
save_and_show("exp2_cold_start_context_bytes.png")


# -----------------------------
# 3. Restart latency
# -----------------------------
replay_restart = []
checkpoint_restart = []

with open(LOG_FILE, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"].startswith("restart"):
            idx = int(r["phase"].split("_")[1])
            if r["agent"] == "replay":
                replay_restart.append((idx, r["latency_ms"]))
            elif r["agent"] == "checkpoint":
                checkpoint_restart.append((idx, r["resume_latency_ms"]))

replay_restart.sort()
checkpoint_restart.sort()

x_replay = [i for i, _ in replay_restart]
y_replay = [v for _, v in replay_restart]

x_checkpoint = [i for i, _ in checkpoint_restart]
y_checkpoint = [v for _, v in checkpoint_restart]

plt.figure()
plt.plot(x_replay, y_replay, marker="o", label="Replay (restart)")
plt.plot(x_checkpoint, y_checkpoint, marker="o", label="Checkpoint (resume)")
plt.xlabel("Restart number")
plt.ylabel("Latency (ms)")
plt.title("Restart Cost")
plt.legend()
save_and_show("exp2_restart_latency.png")


# -----------------------------
# 4. Restart context bytes
# -----------------------------
replay_restart_ctx = []
checkpoint_restart_ctx = []

with open(LOG_FILE, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"].startswith("restart"):
            idx = int(r["phase"].split("_")[1])
            if r["agent"] == "replay":
                replay_restart_ctx.append((idx, r["context_bytes"]))
            elif r["agent"] == "checkpoint":
                checkpoint_restart_ctx.append((idx, r["context_bytes"]))

replay_restart_ctx.sort()
checkpoint_restart_ctx.sort()

x_replay = [i for i, _ in replay_restart_ctx]
y_replay = [v for _, v in replay_restart_ctx]

x_checkpoint = [i for i, _ in checkpoint_restart_ctx]
y_checkpoint = [v for _, v in checkpoint_restart_ctx]

plt.figure()
plt.plot(x_replay, y_replay, marker="o", label="Replay context bytes")
plt.plot(x_checkpoint, y_checkpoint, marker="o", label="Checkpoint context bytes")
plt.xlabel("Restart number")
plt.ylabel("Context size (bytes)")
plt.title("Restart Context Size")
plt.legend()
save_and_show("exp2_restart_context_bytes.png")