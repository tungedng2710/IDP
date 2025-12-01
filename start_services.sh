#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANDRA_DIR="$ROOT_DIR/services/chandra_api"
DOTSOCR_DIR="$ROOT_DIR/services/dotsocr"
DEEPSEEKOCR_DIR="$ROOT_DIR/services/deepseekocr"

NETWORK_NAME="${NETWORK_NAME:-idp-stack}"
CHANDRA_IMAGE="${CHANDRA_IMAGE:-chandra_api}"
DOTSOCR_API_IMAGE="${DOTSOCR_API_IMAGE:-dotsocr_api}"
DEEPSEEKOCR_IMAGE="${DEEPSEEKOCR_IMAGE:-deepseekocr}"
PIPELINE_IMAGE="${PIPELINE_IMAGE:-idp_pipeline}"
DOTSOCR_MODEL_VOLUME="${DOTSOCR_MODEL_VOLUME:-dotsocr-model-cache}"
DEEPSEEKOCR_MODEL_VOLUME="${DEEPSEEKOCR_MODEL_VOLUME:-deepseekocr-model-cache}"

if [[ -n "${CHANDRA_GPU_FLAG+x}" ]]; then
  CHANDRA_GPU_FLAG_USER_SET=1
else
  CHANDRA_GPU_FLAG="--gpus all"
  CHANDRA_GPU_FLAG_USER_SET=0
fi

if [[ -n "${DOTSOCR_VLLM_GPU_FLAG+x}" ]]; then
  DOTSOCR_VLLM_GPU_FLAG_USER_SET=1
else
  DOTSOCR_VLLM_GPU_FLAG="--gpus all"
  DOTSOCR_VLLM_GPU_FLAG_USER_SET=0
fi

if [[ -n "${DEEPSEEKOCR_GPU_FLAG+x}" ]]; then
  DEEPSEEKOCR_GPU_FLAG_USER_SET=1
else
  DEEPSEEKOCR_GPU_FLAG="--gpus all"
  DEEPSEEKOCR_GPU_FLAG_USER_SET=0
fi

DOTSOCR_VISIBLE_GPUS="${DOTSOCR_VISIBLE_GPUS:-1}"
DEEPSEEKOCR_VISIBLE_GPUS="${DEEPSEEKOCR_VISIBLE_GPUS:-0}"
DEEPSEEKOCR_GPU_MEMORY="${DEEPSEEKOCR_GPU_MEMORY:-0.7}"

PIPELINE_BASE_URL="${DOT_OCR_BASE_URL:-http://dots-ocr-api:9667}"
PIPELINE_TABLE_URL="${DOT_OCR_TABLE_URL:-http://chandra-api:9670/chandra/extract}"

VLLM_HEALTH_TIMEOUT="${VLLM_HEALTH_TIMEOUT:-900}"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not found in PATH." >&2
    exit 1
  fi
}

split_words() {
  local value="$1"
  local -n target=$2
  target=()
  if [[ -z "$value" ]]; then
    return
  fi
  # shellcheck disable=SC2206
  target=($value)
}

gpu_runtime_available() {
  local force_cpu="${FORCE_CPU_MODE:-}"
  local force_cpu_lc="${force_cpu,,}"
  if [[ "$force_cpu_lc" == "1" || "$force_cpu_lc" == "true" || "$force_cpu_lc" == "yes" ]]; then
    return 1
  fi

  local runtimes
  if ! runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null); then
    return 1
  fi

  if [[ -n "$runtimes" ]] && command -v nvidia-smi >/dev/null 2>&1 && grep -q '"nvidia"' <<<"$runtimes"; then
    return 0
  fi
  return 1
}

configure_gpu_flags() {
  if gpu_runtime_available; then
    log "Detected NVIDIA GPU runtime; requesting GPU access for services."
    return
  fi

  local force_cpu="${FORCE_CPU_MODE:-}"
  if [[ -n "$force_cpu" ]]; then
    log "FORCE_CPU_MODE=$force_cpu -> starting services without GPU acceleration."
  else
    log "Warning: NVIDIA GPU runtime not detected. Starting services without '--gpus' flags."
    log "Set FORCE_CPU_MODE=1 to skip detection or install NVIDIA Container Toolkit to re-enable GPUs."
  fi

  if (( ! CHANDRA_GPU_FLAG_USER_SET )); then
    CHANDRA_GPU_FLAG=""
  fi
  if (( ! DOTSOCR_VLLM_GPU_FLAG_USER_SET )); then
    DOTSOCR_VLLM_GPU_FLAG=""
  fi
  if (( ! DEEPSEEKOCR_GPU_FLAG_USER_SET )); then
    DEEPSEEKOCR_GPU_FLAG=""
  fi
}

ensure_network() {
  if ! docker network inspect "$1" >/dev/null 2>&1; then
    log "Creating docker network '$1'..."
    docker network create "$1" >/dev/null
  fi
}

build_image() {
  local image="$1"
  local dockerfile="$2"
  local context="$3"
  log "Building image '$image' from $dockerfile ..."
  docker build -t "$image" -f "$dockerfile" "$context"
}

remove_container() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -Eq "^${name}\$"; then
    log "Removing existing container '$name'..."
    docker rm -f "$name" >/dev/null
  fi
}

wait_for_health() {
  local container="$1"
  local timeout="${2:-600}"
  local start
  start=$(date +%s)

  while true; do
    local status
    status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)
    if [[ "$status" == "healthy" ]]; then
      log "'$container' is healthy."
      return 0
    fi

    local now
    now=$(date +%s)
    if (( now - start > timeout )); then
      echo "Warning: Timed out waiting for '$container' to become healthy (last status: ${status:-unknown})." >&2
      return 1
    fi
    sleep 5
  done
}

start_chandra_api() {
  log "Launching chandra_api service..."
  remove_container "chandra-api"
  local -a gpu_args=()
  split_words "$CHANDRA_GPU_FLAG" gpu_args
  docker run -d \
    "${gpu_args[@]}" \
    --name chandra-api \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -p 9670:9670 \
    "$CHANDRA_IMAGE"
}

start_vllm_service() {
  log "Launching VLLM server for dots.ocr..."
  remove_container "vllm-dots-ocr"
  local -a gpu_args=()
  split_words "$DOTSOCR_VLLM_GPU_FLAG" gpu_args
  docker run -d \
    "${gpu_args[@]}" \
    --name vllm-dots-ocr \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    --shm-size=1g \
    -p 8000:8000 \
    -e CUDA_VISIBLE_DEVICES="$DOTSOCR_VISIBLE_GPUS" \
    -v "${DOTSOCR_MODEL_VOLUME}:/root/.cache/huggingface" \
    --health-cmd "curl -f http://localhost:8000/health || exit 1" \
    --health-interval 10s \
    --health-retries 5 \
    --health-timeout 5s \
    --health-start-period 580s \
    --entrypoint vllm \
    vllm/vllm-openai:latest \
    serve rednote-hilab/dots.ocr --trust-remote-code --async-scheduling --gpu-memory-utilization 0.65 --host 0.0.0.0 --port 8000 --mm-processor-cache-gb 0

  if ! wait_for_health "vllm-dots-ocr" "$VLLM_HEALTH_TIMEOUT"; then
    log "Continuing even though 'vllm-dots-ocr' is not healthy yet."
  fi
}

start_dotsocr_api() {
  log "Launching dots.ocr API..."
  remove_container "dots-ocr-api"
  docker run -d \
    --name dots-ocr-api \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -p 9667:9667 \
    -e VLLM_IP=vllm-dots-ocr \
    -e VLLM_PORT=8000 \
    -e MODEL_NAME="rednote-hilab/dots.ocr" \
    -e PYTHON_PATH="/app/dots.ocr:\$PYTHON_PATH" \
    -v "$DOTSOCR_DIR:/app" \
    --health-cmd "curl -f http://localhost:9667/health || exit 1" \
    --health-interval 30s \
    --health-retries 3 \
    --health-timeout 10s \
    --health-start-period 30s \
    "$DOTSOCR_API_IMAGE" \
    sh -c "python api_service.py"
}

start_deepseekocr_api() {
  log "Launching deepseek OCR API..."
  remove_container "deepseekocr-api"
  local -a gpu_args=()
  split_words "$DEEPSEEKOCR_GPU_FLAG" gpu_args
  docker run -d \
    "${gpu_args[@]}" \
    --name deepseekocr-api \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -p 9666:9666 \
    -e CUDA_VISIBLE_DEVICES="$DEEPSEEKOCR_VISIBLE_GPUS" \
    -e GPU_MEMORY_UTILIZATION="$DEEPSEEKOCR_GPU_MEMORY" \
    -v "${DEEPSEEKOCR_MODEL_VOLUME}:/root/.cache/huggingface" \
    -v "$DEEPSEEKOCR_DIR:/app" \
    "$DEEPSEEKOCR_IMAGE" \
    sh -c "python3 api_service.py"
}

start_pipeline_api() {
  log "Launching main pipeline API..."
  remove_container "idp-pipeline"
  docker run -d \
    --name idp-pipeline \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -p 7860:7875 \
    -e DOT_OCR_BASE_URL="$PIPELINE_BASE_URL" \
    -e DOT_OCR_TABLE_URL="$PIPELINE_TABLE_URL" \
    "$PIPELINE_IMAGE"
}

main() {
  require_command docker
  configure_gpu_flags
  ensure_network "$NETWORK_NAME"

  build_image "$CHANDRA_IMAGE" "$CHANDRA_DIR/Dockerfile" "$ROOT_DIR"
  build_image "$DOTSOCR_API_IMAGE" "$DOTSOCR_DIR/Dockerfile" "$DOTSOCR_DIR"
  build_image "$DEEPSEEKOCR_IMAGE" "$DEEPSEEKOCR_DIR/Dockerfile" "$DEEPSEEKOCR_DIR"
  build_image "$PIPELINE_IMAGE" "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

  start_chandra_api
  start_vllm_service
  start_dotsocr_api
  start_deepseekocr_api
  start_pipeline_api

  echo
  log "All services are up:"
  log "- Chandra API        : http://localhost:9670/chandra/extract"
  log "- dots.ocr API       : http://localhost:9667/health"
  log "- DeepSeek OCR API   : http://localhost:9666/health"
  log "- Pipeline API       : http://localhost:7875/health"
  log "Use 'docker ps --filter \"name=idp\" --filter \"name=ocr\"' to verify container status."
}

main "$@"
