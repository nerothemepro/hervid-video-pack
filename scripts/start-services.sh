#!/usr/bin/env bash
# HerVid — start Pipeline API + Orchestrator
# Usage: bash start-services.sh [--restart]

set -euo pipefail

VENV="/workspace/.venvs/hermes-agent/bin/python"
CORE="/workspace/hervid-video-pack/core"
ENV_FILE="/opt/data/hermes/media-pipeline.env"
LOG_DIR="/tmp/hervid-logs"

mkdir -p "$LOG_DIR"

# Load specific keys from env file (safe: avoids sourcing multi-word values)
_env_get() {
  local key="$1" default="$2"
  local val
  val=$(grep -m1 "^${key}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '\r')
  echo "${val:-$default}"
}

export LM_STUDIO_BASE_URL="${LM_STUDIO_BASE_URL:-$(_env_get LM_STUDIO_BASE_URL http://host.docker.internal:1234/v1)}"
export LM_MODEL="${LM_MODEL:-$(_env_get LM_MODEL google/gemma-4-12b-qat)}"
export HVP_API_URL="${HVP_API_URL:-http://localhost:8500}"
export HVP_POLL_INTERVAL="${HVP_POLL_INTERVAL:-30}"
export HVP_MAX_WAIT="${HVP_MAX_WAIT:-7200}"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(_env_get TELEGRAM_BOT_TOKEN '')}"
export HVP_ENV_FILE="$ENV_FILE"
export HVP_PORT="${HVP_PORT:-8500}"
export HVO_PORT="${HVO_PORT:-8501}"

# ---- Stop existing instances if --restart or already running ----
if [ "${1:-}" = "--restart" ] || pgrep -f "pipeline_api.py" > /dev/null 2>&1; then
  echo "[hervid] stopping existing pipeline_api..."
  pkill -f "pipeline_api.py" 2>/dev/null || true
  sleep 2
fi
if [ "${1:-}" = "--restart" ] || pgrep -f "orchestrate.py serve" > /dev/null 2>&1; then
  echo "[hervid] stopping existing orchestrate..."
  pkill -f "orchestrate.py serve" 2>/dev/null || true
  sleep 2
fi

# ---- Start Pipeline API (:8500) ----
echo "[hervid] starting pipeline_api on :${HVP_PORT}..."
nohup "$VENV" "$CORE/pipeline_api.py" \
  > "$LOG_DIR/pipeline_api.log" 2>&1 &
PIPELINE_PID=$!
echo "$PIPELINE_PID" > /tmp/hervid-pipeline.pid

# ---- Wait for pipeline_api to be ready ----
for i in $(seq 1 15); do
  if curl -s "http://localhost:${HVP_PORT}/health" > /dev/null 2>&1; then
    echo "[hervid] pipeline_api ready (pid=$PIPELINE_PID)"
    break
  fi
  sleep 1
done

# ---- Start Orchestrator (:8501) ----
echo "[hervid] starting orchestrator on :${HVO_PORT}..."
nohup "$VENV" "$CORE/orchestrate.py" serve \
  > "$LOG_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "$ORCH_PID" > /tmp/hervid-orchestrator.pid

# ---- Wait for orchestrator to be ready ----
for i in $(seq 1 15); do
  if curl -s "http://localhost:${HVO_PORT}/health" > /dev/null 2>&1; then
    echo "[hervid] orchestrator ready (pid=$ORCH_PID)"
    break
  fi
  sleep 1
done

# ---- Final health check ----
echo ""
echo "=== HerVid service status ==="
curl -s "http://localhost:${HVP_PORT}/health"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  pipeline_api  :${HVP_PORT}  ok={d[\"ok\"]}  comfyui={d[\"comfyui_reachable\"]}')" 2>/dev/null || echo "  pipeline_api  :${HVP_PORT}  NOT READY"
curl -s "http://localhost:${HVO_PORT}/health"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  orchestrator  :${HVO_PORT}  ok={d[\"ok\"]}  lm={d[\"lm_studio\"]}  pipeline={d[\"pipeline_api\"]}')" 2>/dev/null || echo "  orchestrator  :${HVO_PORT}  NOT READY"
echo ""
echo "Logs: $LOG_DIR/"
echo "Stop: bash $(dirname "$0")/stop-services.sh"
