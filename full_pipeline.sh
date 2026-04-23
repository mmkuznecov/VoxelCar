#!/usr/bin/env bash
# full_pipeline.sh — end-to-end voxel-car pipeline
#
# Stages: generate → preprocess → train OccNet → eval OccNet → train PPO → eval PPO
# Run from repo root with venv active: ./full_pipeline.sh
#
# Skip any stage:  SKIP_GENERATE=1 SKIP_PREPROCESS=1 ./full_pipeline.sh
#
# Key knobs (env vars):
#   N_SAMPLES, OCC_EPOCHS, OCC_BATCH_SIZE, RL_STEPS, RL_N_ENVS, RL_PRESET,
#   EVAL_OCC_N, EVAL_OCC_PRESET, EVAL_RL_ORACLE_N, EVAL_RL_CL_N
#
# WARNING: 15k runs × 40 frames ≈ 46 GB. Full pipeline: 12-24+ hours.

set -euo pipefail

# --- Stage defaults ---
N_SAMPLES="${N_SAMPLES:-15000}"
BASE_SEED_GEN="${BASE_SEED_GEN:-100}"
N_JOBS="${N_JOBS:--1}"
OCC_EPOCHS="${OCC_EPOCHS:-10}"
OCC_BATCH_SIZE="${OCC_BATCH_SIZE:-128}"
OCC_LR="${OCC_LR:-1e-3}"
EVAL_OCC_N="${EVAL_OCC_N:-20}"
EVAL_OCC_PRESET="${EVAL_OCC_PRESET:-winding}"
RL_STEPS="${RL_STEPS:-1000000}"
RL_N_ENVS="${RL_N_ENVS:-8}"
RL_PRESET="${RL_PRESET:-easy}"
EVAL_RL_ORACLE_N="${EVAL_RL_ORACLE_N:-100}"
EVAL_RL_CL_N="${EVAL_RL_CL_N:-20}"

# --- Paths ---
DATA_DIR="./data/dataset"
PP_DIR="./data/preprocessed"
OCC_OUT_ROOT="./output/OccModel"
OCC_RUN_DIR="${OCC_OUT_ROOT}/occnet_main"
OCC_CKPT="${OCC_RUN_DIR}/ckpt_best.pt"
RL_OUT_ROOT="./output/RLModel"
RL_RUN_DIR="${RL_OUT_ROOT}/ppo_main"
RL_POLICY="${RL_RUN_DIR}/ppo_voxel_car_final.zip"
CL_EVAL_DIR="./output/closed_loop_eval"
CL_RL_EVAL_DIR="./output/closed_loop_rl_eval"

# --- Helpers ---
mkdir -p data output
stage() { echo ""; echo "==== $1 ($(date '+%H:%M:%S')) ===="; }
fail()  { echo "ERROR: $1" >&2; exit 1; }
skip()  { [[ "${!1:-0}" == "1" ]]; }

PIPELINE_T0=$(date +%s)

# 1. Generate
if skip SKIP_GENERATE; then stage "[1/6] SKIPPED"; else
    stage "[1/6] Generating ${N_SAMPLES} scenarios → ${DATA_DIR}"
    python generate.py --out-dir "${DATA_DIR}" --n "${N_SAMPLES}" \
        --base-seed "${BASE_SEED_GEN}" --n-jobs "${N_JOBS}"
fi
[[ -f "${DATA_DIR}/manifest.json" ]] || fail "manifest.json missing — did generation succeed?"

# 2. Preprocess
if skip SKIP_PREPROCESS; then stage "[2/6] SKIPPED"; else
    stage "[2/6] Preprocessing → ${PP_DIR}"
    python preprocess.py --data-dir "${DATA_DIR}" --out-dir "${PP_DIR}"
fi
[[ -f "${PP_DIR}/index.json" ]] || fail "index.json missing — did preprocessing succeed?"

# 3. Train OccNet
if skip SKIP_TRAIN_OCC; then stage "[3/6] SKIPPED"; else
    stage "[3/6] Training OccNet (${OCC_EPOCHS} epochs, batch ${OCC_BATCH_SIZE}) → ${OCC_RUN_DIR}"
    python train.py --data-root "${PP_DIR}" --out-dir "${OCC_OUT_ROOT}" \
        --run-name "occnet_main" --epochs "${OCC_EPOCHS}" \
        --batch-size "${OCC_BATCH_SIZE}" --lr "${OCC_LR}"
fi
[[ -f "${OCC_CKPT}" ]] || fail "OccNet checkpoint missing: ${OCC_CKPT}"

# 4. Eval OccNet (closed-loop, A*)
if skip SKIP_EVAL_OCC; then stage "[4/6] SKIPPED"; else
    stage "[4/6] Closed-loop OccNet+A* (n=${EVAL_OCC_N}, preset=${EVAL_OCC_PRESET}) → ${CL_EVAL_DIR}"
    python run_closed_loop.py --ckpt "${OCC_CKPT}" --out-dir "${CL_EVAL_DIR}" \
        --n "${EVAL_OCC_N}" --preset "${EVAL_OCC_PRESET}"
fi

# 5. Train PPO
if skip SKIP_TRAIN_RL; then stage "[5/6] SKIPPED"; else
    stage "[5/6] Training PPO (${RL_STEPS} steps, ${RL_N_ENVS} envs, preset=${RL_PRESET}) → ${RL_RUN_DIR}"
    python train_rl.py --out-dir "${RL_OUT_ROOT}" --run-name "ppo_main" \
        --preset "${RL_PRESET}" --n-envs "${RL_N_ENVS}" --subproc --steps "${RL_STEPS}"
fi
[[ -f "${RL_POLICY}" ]] || fail "RL policy missing: ${RL_POLICY}"

# 6. Eval PPO
if skip SKIP_EVAL_RL; then stage "[6/6] SKIPPED"; else
    stage "[6a/6] Oracle RL eval (n=${EVAL_RL_ORACLE_N})"
    python eval_rl.py --model "${RL_POLICY}" --preset "${RL_PRESET}" \
        --n "${EVAL_RL_ORACLE_N}" --out "${RL_RUN_DIR}/rl_eval.json"

    stage "[6b/6] Closed-loop camera→OccNet→PPO (n=${EVAL_RL_CL_N}) → ${CL_RL_EVAL_DIR}"
    python run_closed_loop_rl.py --ckpt "${OCC_CKPT}" --rl-policy "${RL_POLICY}" \
        --out-dir "${CL_RL_EVAL_DIR}" --n "${EVAL_RL_CL_N}" --preset "${RL_PRESET}"
fi

# Summary
TOTAL_SEC=$(( $(date +%s) - PIPELINE_T0 ))
echo ""
echo "==== Pipeline complete in $(( TOTAL_SEC/3600 ))h $(( (TOTAL_SEC%3600)/60 ))m ($(date '+%H:%M:%S')) ===="
echo "  OccNet ckpt:          ${OCC_CKPT}"
echo "  OccNet eval report:   ${OCC_RUN_DIR}/eval_report.json"
echo "  Closed-loop (A*):     ${CL_EVAL_DIR}/<timestamp>/summary.json"
echo "  PPO policy:           ${RL_POLICY}"
echo "  PPO oracle eval:      ${RL_RUN_DIR}/rl_eval.json"
echo "  PPO closed-loop eval: ${CL_RL_EVAL_DIR}/<timestamp>/summary.json"