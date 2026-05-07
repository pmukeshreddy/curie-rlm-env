#!/usr/bin/env bash
# One-command rollout-only debug:
#   1. Starts vLLM (uses prime-rl's bundled vllm) serving Qwen3-8B
#   2. Waits for /v1/models to respond
#   3. Runs scripts/debug_rollouts.py against it
#   4. Kills vLLM on exit (whether success or failure)
#
# Usage:
#   bash scripts/debug_rollouts.sh
#
# Override defaults via env vars:
#   PRIME_RL_PROJECT=/path/to/prime-rl   (default /workspace/prime-rl)
#   MODEL_NAME=Qwen/Qwen3-8B
#   VLLM_PORT=8000
#   VLLM_LOG=/tmp/vllm.log
#
# All vLLM stdout/stderr → $VLLM_LOG (default /tmp/vllm.log).
# Rollout debug stdout → terminal + results/debug_rollouts/run.stdout.
set -euo pipefail

PRIME_RL_PROJECT="${PRIME_RL_PROJECT:-/workspace/prime-rl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm.log}"
BASE_URL="http://localhost:${VLLM_PORT}/v1"

mkdir -p results/debug_rollouts

echo "[wrapper] Starting vLLM: ${MODEL_NAME} on port ${VLLM_PORT}"
echo "[wrapper] vLLM log → ${VLLM_LOG}"

uv run --project "${PRIME_RL_PROJECT}" \
    vllm serve "${MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo ""
    echo "[wrapper] Cleaning up: killing vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[wrapper] Waiting for vLLM ${BASE_URL} to respond (up to 5 min)..."
READY=0
for i in $(seq 1 150); do
    if curl -fs "${BASE_URL}/models" > /dev/null 2>&1; then
        echo "[wrapper] vLLM ready after ${i}*2s"
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[wrapper] ERROR: vLLM process died before ready. Last 40 lines of ${VLLM_LOG}:"
        tail -n 40 "${VLLM_LOG}" || true
        exit 1
    fi
    sleep 2
done

if [ "${READY}" -ne 1 ]; then
    echo "[wrapper] ERROR: vLLM did not become ready in 5 min. Last 40 lines of ${VLLM_LOG}:"
    tail -n 40 "${VLLM_LOG}" || true
    exit 1
fi

echo "[wrapper] Running scripts/debug_rollouts.py"
echo "==============================================================================="

PYTHONPATH=/workspace/curie-rlm-env/src \
  uv run --project "${PRIME_RL_PROJECT}" \
  python scripts/debug_rollouts.py --base-url "${BASE_URL}" --model "${MODEL_NAME}" \
  | tee results/debug_rollouts/run.stdout

echo "==============================================================================="
echo "[wrapper] Done. Summary above; trajectories above; sandbox/env stderr in:"
echo "          results/debug_rollouts/example_*.stderr.log"
