#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash minimal_rap_bridge/run_native_policy_pipeline.sh --native-video /path/to/native_agent.mp4 [options]

Required:
  --native-video PATH          Native visualize output video (.mp4)

Optional:
  --python-bin PATH            Python executable (default: ./.venv-pufferdrive-rap/bin/python if exists, else python)
  --out-root DIR               Root output dir (default: /tmp/pd_rap_pipeline_native)
  --frames N                   Number of frames (default: 80)
  --episode-length N           Episode length (default: 120)
  --seed N                     Seed (default: 1)
  --map-dir DIR                Map binaries dir (default: resources/drive/binaries)
  --num-maps N                 Number of maps (default: 1)
  --num-agents N               Number of agents (default: 21; recommended single-env)
  --cameras LIST               Comma-separated cameras (default: CAM_F0,CAM_L0,CAM_R0)
  --panel-camera ID            Panel camera and matching camera (default: CAM_F0)
  --ego-select-mode MODE       index|native_visualize|random_seeded (default: random_seeded)
  --ego-agent-index N          Ego index when --ego-select-mode index (default: 6)
  --ego-random-seed N          Seed for random_seeded ego mode (default: --seed)
  --native-ego-log PATH        Parse NATIVE_EGO_* metadata from native visualize log and override ego selection
  --control-mode MODE          control_vehicles|control_agents|control_wosac|control_sdc_only (default: control_vehicles)
  --init-mode MODE             create_all_valid|create_only_controlled (default: create_all_valid)
  --include-ego-box            Include ego box (default: enabled)
  --no-include-ego-box         Disable ego box
  --policy-path PATH           Policy weights path (default: resources/drive/puffer_drive_weights.bin)
  --strip-cameras LIST         Cameras for RAP strip video (default: CAM_L0,CAM_F0,CAM_R0)
  --strip-fps N                RAP strip video fps (default: 10)
  --strip-backend MODE         RAP strip backend auto|ffmpeg|opencv (default: auto)
  --strip-codec FOURCC         RAP strip OpenCV FOURCC fallback (default: MJPG)
  --strip-video PATH           RAP strip video output path (default: <out-root>/rap_strip_lfr.mp4)
  --skip-strip-video           Skip RAP strip video generation
  --match-native-length        Clamp RAP frames to native video length (default: enabled)
  --no-match-native-length     Disable native-length clamping
  --frame-offset N             Native-vs-RAP offset (default: 0)
  --sample-every N             Native-vs-RAP sample stride (default: 1)
  --max-samples N              Native-vs-RAP max pairs (default: 80)
  --skip-action-check          Skip native-reference vs RAP action log comparison
EOF
}

PYTHON_BIN=""
OUT_ROOT="/tmp/pd_rap_pipeline_native"
NATIVE_VIDEO=""
FRAMES=80
EPISODE_LENGTH=120
SEED=1
MAP_DIR="resources/drive/binaries"
NUM_MAPS=1
NUM_AGENTS=21
CAMERAS="CAM_F0,CAM_L0,CAM_R0"
PANEL_CAMERA="CAM_F0"
EGO_SELECT_MODE="random_seeded"
EGO_AGENT_INDEX=6
EGO_RANDOM_SEED=""
NATIVE_EGO_LOG=""
CONTROL_MODE="control_vehicles"
INIT_MODE="create_all_valid"
INCLUDE_EGO_BOX=1
POLICY_PATH="resources/drive/puffer_drive_weights.bin"
STRIP_CAMERAS="CAM_L0,CAM_F0,CAM_R0"
STRIP_FPS=10
STRIP_BACKEND="auto"
STRIP_CODEC="MJPG"
STRIP_VIDEO=""
DO_STRIP_VIDEO=1
MATCH_NATIVE_LENGTH=1
FRAME_OFFSET=0
SAMPLE_EVERY=1
MAX_SAMPLES=80
DO_ACTION_CHECK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --native-video) NATIVE_VIDEO="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --episode-length) EPISODE_LENGTH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --map-dir) MAP_DIR="$2"; shift 2 ;;
    --num-maps) NUM_MAPS="$2"; shift 2 ;;
    --num-agents) NUM_AGENTS="$2"; shift 2 ;;
    --cameras) CAMERAS="$2"; shift 2 ;;
    --panel-camera) PANEL_CAMERA="$2"; shift 2 ;;
    --ego-select-mode) EGO_SELECT_MODE="$2"; shift 2 ;;
    --ego-agent-index) EGO_AGENT_INDEX="$2"; shift 2 ;;
    --ego-random-seed) EGO_RANDOM_SEED="$2"; shift 2 ;;
    --native-ego-log) NATIVE_EGO_LOG="$2"; shift 2 ;;
    --control-mode) CONTROL_MODE="$2"; shift 2 ;;
    --init-mode) INIT_MODE="$2"; shift 2 ;;
    --include-ego-box) INCLUDE_EGO_BOX=1; shift 1 ;;
    --no-include-ego-box) INCLUDE_EGO_BOX=0; shift 1 ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;;
    --strip-cameras) STRIP_CAMERAS="$2"; shift 2 ;;
    --strip-fps) STRIP_FPS="$2"; shift 2 ;;
    --strip-backend) STRIP_BACKEND="$2"; shift 2 ;;
    --strip-codec) STRIP_CODEC="$2"; shift 2 ;;
    --strip-video) STRIP_VIDEO="$2"; shift 2 ;;
    --skip-strip-video) DO_STRIP_VIDEO=0; shift 1 ;;
    --match-native-length) MATCH_NATIVE_LENGTH=1; shift 1 ;;
    --no-match-native-length) MATCH_NATIVE_LENGTH=0; shift 1 ;;
    --frame-offset) FRAME_OFFSET="$2"; shift 2 ;;
    --sample-every) SAMPLE_EVERY="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-action-check) DO_ACTION_CHECK=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${NATIVE_VIDEO}" ]]; then
  echo "Missing required --native-video" >&2
  usage
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.venv-pufferdrive-rap/bin/python" ]]; then
    PYTHON_BIN="./.venv-pufferdrive-rap/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "${EGO_RANDOM_SEED}" ]]; then
  EGO_RANDOM_SEED="${SEED}"
fi

if [[ -n "${NATIVE_EGO_LOG}" ]]; then
  if [[ ! -f "${NATIVE_EGO_LOG}" ]]; then
    echo "Native ego log not found: ${NATIVE_EGO_LOG}" >&2
    exit 1
  fi

  NATIVE_EGO_SLOT="$(awk -F= '/^NATIVE_EGO_SLOT=/{print $2}' "${NATIVE_EGO_LOG}" | tail -n 1)"
  NATIVE_EGO_ID="$(awk -F= '/^NATIVE_EGO_ID=/{print $2}' "${NATIVE_EGO_LOG}" | tail -n 1)"
  NATIVE_EGO_RANDOM_SEED="$(awk -F= '/^NATIVE_EGO_RANDOM_SEED=/{print $2}' "${NATIVE_EGO_LOG}" | tail -n 1)"

  if [[ -n "${NATIVE_EGO_SLOT}" ]]; then
    if [[ ! "${NATIVE_EGO_SLOT}" =~ ^-?[0-9]+$ ]]; then
      echo "Invalid NATIVE_EGO_SLOT in ${NATIVE_EGO_LOG}: ${NATIVE_EGO_SLOT}" >&2
      exit 1
    fi
    EGO_SELECT_MODE="index"
    EGO_AGENT_INDEX="${NATIVE_EGO_SLOT}"
    echo "[info] using native ego slot from log: slot=${NATIVE_EGO_SLOT}${NATIVE_EGO_ID:+, id=${NATIVE_EGO_ID}}"
  elif [[ -n "${NATIVE_EGO_RANDOM_SEED}" && "${NATIVE_EGO_RANDOM_SEED}" =~ ^-?[0-9]+$ && "${NATIVE_EGO_RANDOM_SEED}" -ge 0 ]]; then
    EGO_SELECT_MODE="random_seeded"
    EGO_RANDOM_SEED="${NATIVE_EGO_RANDOM_SEED}"
    echo "[info] using native ego random seed from log: seed=${NATIVE_EGO_RANDOM_SEED}"
  else
    echo "No usable NATIVE_EGO_SLOT or NATIVE_EGO_RANDOM_SEED found in ${NATIVE_EGO_LOG}" >&2
    exit 1
  fi
fi

if [[ "${MATCH_NATIVE_LENGTH}" -eq 1 ]]; then
  NATIVE_FRAME_COUNT="$(
    "${PYTHON_BIN}" - "${NATIVE_VIDEO}" <<'PY'
import cv2
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"failed to open native video: {video_path}")
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(count)
PY
  )"
  if [[ -z "${NATIVE_FRAME_COUNT}" || "${NATIVE_FRAME_COUNT}" -le 0 ]]; then
    echo "Failed to determine native video frame count for: ${NATIVE_VIDEO}" >&2
    exit 1
  fi

  EFFECTIVE_MAX_FRAMES="${NATIVE_FRAME_COUNT}"
  if [[ "${FRAME_OFFSET}" -gt 0 ]]; then
    EFFECTIVE_MAX_FRAMES=$((NATIVE_FRAME_COUNT - FRAME_OFFSET))
  fi
  if [[ "${EFFECTIVE_MAX_FRAMES}" -lt 1 ]]; then
    echo "No overlapping frames after frame offset: native_frames=${NATIVE_FRAME_COUNT}, frame_offset=${FRAME_OFFSET}" >&2
    exit 1
  fi
  if [[ "${FRAMES}" -gt "${EFFECTIVE_MAX_FRAMES}" ]]; then
    echo "[info] clamping --frames from ${FRAMES} to ${EFFECTIVE_MAX_FRAMES} to match native video horizon"
    FRAMES="${EFFECTIVE_MAX_FRAMES}"
  fi
  if [[ "${MAX_SAMPLES}" -gt "${FRAMES}" ]]; then
    MAX_SAMPLES="${FRAMES}"
  fi
  echo "[info] native length guard: native_frames=${NATIVE_FRAME_COUNT}, frame_offset=${FRAME_OFFSET}, run_frames=${FRAMES}"
fi

NUM_ENVS="$(
  PD_NUM_AGENTS="${NUM_AGENTS}" \
  PD_NUM_MAPS="${NUM_MAPS}" \
  PD_MAP_DIR="${MAP_DIR}" \
  PD_CONTROL_MODE="${CONTROL_MODE}" \
  PD_INIT_MODE="${INIT_MODE}" \
  PD_EPISODE_LENGTH="${EPISODE_LENGTH}" \
  PD_SEED="${SEED}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pufferlib.ocean.drive.drive import Drive

env = Drive(
    num_agents=int(os.environ["PD_NUM_AGENTS"]),
    num_maps=int(os.environ["PD_NUM_MAPS"]),
    map_dir=os.environ["PD_MAP_DIR"],
    control_mode=os.environ["PD_CONTROL_MODE"],
    init_mode=os.environ["PD_INIT_MODE"],
    episode_length=int(os.environ["PD_EPISODE_LENGTH"]),
    resample_frequency=0,
    report_interval=100,
)
env.reset(seed=int(os.environ["PD_SEED"]))
print(f"num_envs={env.num_envs}")
env.close()
PY
)"
NUM_ENVS="$(printf '%s\n' "${NUM_ENVS}" | awk -F= '/^num_envs=/{print $2}' | tail -n 1)"
if [[ -z "${NUM_ENVS}" ]]; then
  echo "Failed to determine Drive num_envs for guard check." >&2
  exit 1
fi
if [[ "${NUM_ENVS}" -gt 1 ]]; then
  echo "Refusing to run: this config creates num_envs=${NUM_ENVS} (>1), which merges multiple simulations and breaks native-vs-RAP parity." >&2
  echo "Use a single-env setting (for this map/config, --num-agents 21 works)." >&2
  exit 1
fi

STATE_COMPARE_DIR="${OUT_ROOT}/state_compare"
RAP_DIR="${OUT_ROOT}/rap_frames"
MATCH_DIR="${OUT_ROOT}/native_vs_rap"
REF_ACTION_LOG="${OUT_ROOT}/native_policy_actions_ref.csv"
RAP_ACTION_LOG="${OUT_ROOT}/native_policy_actions_rap.csv"
ACTION_REPORT="${OUT_ROOT}/action_log_compare.txt"
if [[ -z "${STRIP_VIDEO}" ]]; then
  STRIP_VIDEO="${OUT_ROOT}/rap_strip_lfr.mp4"
fi

mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --frames "${FRAMES}"
  --episode-length "${EPISODE_LENGTH}"
  --seed "${SEED}"
  --map-dir "${MAP_DIR}"
  --num-maps "${NUM_MAPS}"
  --num-agents "${NUM_AGENTS}"
  --cameras "${CAMERAS}"
  --control-mode "${CONTROL_MODE}"
  --init-mode "${INIT_MODE}"
  --ego-select-mode "${EGO_SELECT_MODE}"
  --ego-agent-index "${EGO_AGENT_INDEX}"
  --ego-random-seed "${EGO_RANDOM_SEED}"
  --control-source native_policy
  --policy-path "${POLICY_PATH}"
)

if [[ "${INCLUDE_EGO_BOX}" -eq 1 ]]; then
  COMMON_ARGS+=(--include-ego-box)
fi

if [[ "${DO_ACTION_CHECK}" -eq 1 ]]; then
  echo "[step] Native policy reference actions -> ${REF_ACTION_LOG}"
  "${PYTHON_BIN}" minimal_rap_bridge/log_native_policy_actions_reference.py \
    --out-csv "${REF_ACTION_LOG}" \
    --frames "${FRAMES}" \
    --episode-length "${EPISODE_LENGTH}" \
    --seed "${SEED}" \
    --map-dir "${MAP_DIR}" \
    --num-maps "${NUM_MAPS}" \
    --num-agents "${NUM_AGENTS}" \
    --control-mode "${CONTROL_MODE}" \
    --init-mode "${INIT_MODE}" \
    --ego-select-mode "${EGO_SELECT_MODE}" \
    --ego-agent-index "${EGO_AGENT_INDEX}" \
    --ego-random-seed "${EGO_RANDOM_SEED}" \
    --policy-path "${POLICY_PATH}"
fi

echo "[step] BEV+RAP compare (native_policy) -> ${STATE_COMPARE_DIR}"
"${PYTHON_BIN}" minimal_rap_bridge/compare_state_vs_rap_baseline.py \
  --out-dir "${STATE_COMPARE_DIR}" \
  --panel-camera "${PANEL_CAMERA}" \
  "${COMMON_ARGS[@]}"

echo "[step] Pure RAP render (native_policy) -> ${RAP_DIR}"
"${PYTHON_BIN}" minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py \
  --out-dir "${RAP_DIR}" \
  --actions-log "${RAP_ACTION_LOG}" \
  "${COMMON_ARGS[@]}"

if [[ "${DO_STRIP_VIDEO}" -eq 1 ]]; then
  echo "[step] RAP strip video (horizontal) -> ${STRIP_VIDEO}"
  "${PYTHON_BIN}" minimal_rap_bridge/make_rap_strip_video.py \
    --rap-dir "${RAP_DIR}" \
    --out-video "${STRIP_VIDEO}" \
    --cameras "${STRIP_CAMERAS}" \
    --fps "${STRIP_FPS}" \
    --backend "${STRIP_BACKEND}" \
    --opencv-codec "${STRIP_CODEC}"
fi

echo "[step] Native-vs-RAP matching -> ${MATCH_DIR}"
"${PYTHON_BIN}" minimal_rap_bridge/compare_native_video_to_rap.py \
  --native-video "${NATIVE_VIDEO}" \
  --rap-dir "${RAP_DIR}" \
  --camera "${PANEL_CAMERA}" \
  --out-dir "${MATCH_DIR}" \
  --frame-offset "${FRAME_OFFSET}" \
  --sample-every "${SAMPLE_EVERY}" \
  --max-samples "${MAX_SAMPLES}"

if [[ "${DO_ACTION_CHECK}" -eq 1 ]]; then
  echo "[step] Compare action logs -> ${ACTION_REPORT}"
  "${PYTHON_BIN}" minimal_rap_bridge/compare_action_logs.py \
    --ref-log "${REF_ACTION_LOG}" \
    --rap-log "${RAP_ACTION_LOG}" \
    --out-report "${ACTION_REPORT}"
fi

cat <<EOF
[done]
- state compare: ${STATE_COMPARE_DIR}
- rap frames: ${RAP_DIR}
- native-vs-rap: ${MATCH_DIR}
EOF
if [[ "${DO_STRIP_VIDEO}" -eq 1 ]]; then
cat <<EOF
- rap strip video: ${STRIP_VIDEO}
EOF
fi

if [[ "${DO_ACTION_CHECK}" -eq 1 ]]; then
cat <<EOF
- ref actions: ${REF_ACTION_LOG}
- rap actions: ${RAP_ACTION_LOG}
- action report: ${ACTION_REPORT}
EOF
fi
