#!/bin/bash

# Two-node profiling script for Qwen3-32B (dense) weight transfer.
#
# Same structure as run-qwen3-30B-A3B-4node-profile.sh, scaled to 2 nodes and
# adapted for a dense model: one train node (8 GPUs) + one rollout node
# (8 GPUs), no expert parallelism, no DP attention.
#
# Usage:
#   bash run-qwen3-32B-2node-profile.sh <MODE> <NODE_RANK> <HEAD_NODE_IP>
#
#   MODE          : broadcast | p2p | shard-delta
#   NODE_RANK     : 0 (head node) | 1 (worker node)
#   HEAD_NODE_IP  : IP address of the head node

set -ex

export PYTHONBUFFERED=16

if [ $# -lt 3 ]; then
    echo "Usage: $0 <MODE> <NODE_RANK> <HEAD_NODE_IP>"
    exit 1
fi

MODE="$1"
NODE_RANK="$2"
HEAD_NODE_IP="$3"

# ---------------------------------------------------------------------------
# Cleanup stale processes (head node only)
# ---------------------------------------------------------------------------
if [ "$NODE_RANK" -eq 0 ]; then
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 python || true
    sleep 3
    pkill -9 ray || true
    pkill -9 python || true
    pkill -9 redis || true
fi

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
NNODES=2
GPUS_PER_NODE=8
NUM_TRAIN_GPUS=8      # 1 node
NUM_ROLLOUT_GPUS=8    # 1 node
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
BUCKET_SIZE_GB="${BUCKET_SIZE_GB:-1.0}"
NO_SAVE_OPTIM=0

NUM_TRAIN_NODES=$((NUM_TRAIN_GPUS / GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# NVLink detection
# ---------------------------------------------------------------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen3-32B"
MODEL_TYPE="qwen3-32B"

MILES_ROOT="/root/miles"
source "${MILES_ROOT}/scripts/models/${MODEL_TYPE}.sh"

export MODEL_ARGS_ROTARY_BASE=1000000

MODES=("$MODE")

run_mode() {
    local mode="$1"

    CKPT_ARGS=(
        --hf-checkpoint "/root/models/${MODEL_NAME}/"
        --ref-load "/root/multinode/${MODEL_NAME}_torch_dist/"
    )
    if [ "$NO_SAVE_OPTIM" -eq 1 ]; then
        CKPT_ARGS+=(--no-save-optim)
    fi

    ROLLOUT_ARGS=(
        --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type deepscaler
        --num-rollout 13
        --rollout-batch-size 4
        --n-samples-per-prompt 4
        --rollout-max-response-len 100
        --rollout-temperature 0.8
        --global-batch-size 16
        --balance-data
    )

    EVAL_ARGS=(
        --eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl
        --n-samples-per-eval-prompt 16
        --eval-max-response-len 16384
        --eval-top-p 0.7
    )

    # dense model: no expert/expert-tensor parallelism
    PERF_ARGS=(
        --tensor-model-parallel-size 8
        --sequence-parallel
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 2048
    )

    GRPO_ARGS=(
        --advantage-estimator gspo
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --entropy-coef 0.00
        --eps-clip 4e-4
    )

    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr 1e-6
        --lr-decay-style constant
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )

    WANDB_ARGS=(
        #--use-wandb
    )

    # one engine on 8 GPUs; dense model so no ep/dp-attention
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
        --sglang-mem-fraction-static 0.8
        --sglang-cuda-graph-bs 1 2 4 8 16
    )
    if [ "$mode" = "p2p" ]; then
        SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
    fi
    if [ "$SKIP_VALIDATION" -eq 1 ]; then
        SGLANG_ARGS+=(--sglang-load-format dummy)
    else
        SGLANG_ARGS+=(--sglang-model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}')
    fi

    BUFFER_SIZE=$((1 * 1024 * 1024 * 1024))

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --actor-num-nodes ${NUM_TRAIN_NODES}
        --actor-num-gpus-per-node ${GPUS_PER_NODE}
        --update-weight-buffer-size ${BUFFER_SIZE}
    )
    if [ "$SKIP_VALIDATION" -eq 0 ]; then
        MISC_ARGS+=(--check-weight-update-equal)
    fi
    if [ "$mode" = "p2p" ]; then
        MISC_ARGS+=(--update-weight-transfer-mode p2p)
    else
        MISC_ARGS+=(--update-weight-transfer-mode broadcast)
    fi

    if [ "$NODE_RANK" -gt 0 ]; then
        sleep 20
    fi

    MC_TRANSFER_TIMEOUT=300
    NCCL_NVLS_VAL="1"

    if [ "$NODE_RANK" -eq 0 ]; then
        ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus ${GPUS_PER_NODE} \
            --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    else
        ray start --address="${HEAD_NODE_IP}:6379" --num-gpus ${GPUS_PER_NODE} --disable-usage-stats
    fi

    RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"MC_TRANSFER_TIMEOUT\": \"${MC_TRANSFER_TIMEOUT}\",
    \"RAY_DEBUG\": \"1\",
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_VAL}\",
    \"MILES_LOG_DIR\": \"${MILES_LOG_DIR:-}\"
  }
}"

    EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Waiting for ${EXPECTED_GPUS} GPUs in Ray cluster..."
        while true; do
            AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null || echo 0)
            echo "  ... detected ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs"
            if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
                break
            fi
            sleep 5
        done
        echo "All ${EXPECTED_GPUS} GPUs available. Submitting job."
    fi

    if [ "$NODE_RANK" -eq 0 ]; then
        ray job submit --address="http://127.0.0.1:8265" \
            --runtime-env-json="${RUNTIME_ENV_JSON}" \
            -- python3 train.py \
            ${MODEL_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${EVAL_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} \
            ${GRPO_ARGS[@]} \
            ${WANDB_ARGS[@]} \
            ${PERF_ARGS[@]} \
            ${SGLANG_ARGS[@]} \
            ${MISC_ARGS[@]}
    fi
}

echo ""
echo "============================================================"
echo "  Running: qwen3-32B dense / ${MODE}"
echo "============================================================"
echo ""

run_mode "$MODE"

echo "Done."
