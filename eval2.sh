#!/bin/bash
set -e
echo "Running eval2.sh..."

# Variables
EVAL_MODEL_BASE=./ckpts/kd_mixin/no_pt
EVAL_OUT_BASE=./eval_result/no_pt
DATASET_PATH=./data/sft_spider_dev_text2sql.json
SIC_PATH=./data_zip/sic_ckpts/sic_spider

# Alpha values and corresponding checkpoint numbers
ALPHA_VALUES=(5e-5 5e-6 5e-5 5e-6)
CHECKPOINT_NUMBERS=(204 204 272 272)

# Associative array to map alpha values to checkpoint numbers
declare -A ALPHA_CHECKPOINT_MAP

# Populate the associative array
for i in "${!ALPHA_VALUES[@]}"; do
    key="${ALPHA_VALUES[$i]}_${CHECKPOINT_NUMBERS[$i]}"
    ALPHA_CHECKPOINT_MAP[$key]=${CHECKPOINT_NUMBERS[$i]}
done

# Ensure evaluation output directory exists
mkdir -p $EVAL_OUT_BASE

# Function to run evaluation
run_evaluation() {
    local START_ALPHA=$1
    local CHECKPOINT_NUMBER=$2
    local CUDA_DEVICE=$3

    # revkl/bird/_alpha_0.2_cos_normal_beta_0.5_const_t2
    # new/bird_5e-5_alpha_0.2_cos_normal_beta_0.5_const_t2
    local EVAL_MODEL=${EVAL_MODEL_BASE}/${START_ALPHA}_alpha_0.2_cos_normal_beta_0.5_const_t2/ckpt-${CHECKPOINT_NUMBER}
    local EVAL_OUT=${EVAL_OUT_BASE}/${START_ALPHA}_alpha_0.2_cos_normal_beta_0.5_const_t2_ckpt-${CHECKPOINT_NUMBER}.txt

    echo "Evaluating model: $EVAL_MODEL"
    echo "Output file: $EVAL_OUT"

    mkdir -p $EVAL_OUT_BASE

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u text2sql_zero_shot.py --llm_path $EVAL_MODEL \
        --dataset_path $DATASET_PATH \
        --sic_path $SIC_PATH \
        --table_num 6 --column_num 10 \
        --max_tokens 4096 --max_new_tokens 256 >> $EVAL_OUT
}

# Split the keys into two arrays for each GPU
KEYS=("${!ALPHA_CHECKPOINT_MAP[@]}")
HALF=$((${#KEYS[@]} / 2))
KEYS_GPU0=("${KEYS[@]:0:$HALF}")
KEYS_GPU1=("${KEYS[@]:$HALF}")

# Run evaluations on GPU 0
for key in "${KEYS_GPU0[@]}"; do
    IFS='_' read -r START_ALPHA CHECKPOINT_NUMBER <<< "$key"
    run_evaluation "$START_ALPHA" "${ALPHA_CHECKPOINT_MAP[$key]}" 3
done

# Run evaluations on GPU 1
for key in "${KEYS_GPU1[@]}"; do
    IFS='_' read -r START_ALPHA CHECKPOINT_NUMBER <<< "$key"
    run_evaluation "$START_ALPHA" "${ALPHA_CHECKPOINT_MAP[$key]}" 3
done

wait