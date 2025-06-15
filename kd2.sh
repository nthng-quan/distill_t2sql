set -e

MODEL_KD=seeklhy/codes-1b-spider
KD_DIR_BASE=./ckpts/kd_mixin/revkl/beta/
LOGDIR_BASE=./train_logs/kd_mixin/revkl/beta/

EPOCHS=4
BS=4
ALPHA_SCHEDULER=cos
MIN_ALPHA=0
ALPHA_STEP_SIZE=3
ALPHA_WARMUP_RATIO=1
LR_WARMUP_RATIO=0.125
LR=5e-5
KD_LOSS=rev_kl
IMG_WEIGHT=-1

TEXT2SQL_DATA_DIR=./data_zip/data/sft_spider_train_text2sql.json
CUDA_DEVICE=2

run_training () {
    local START_ALPHA=$1
    local ALPHA_SCHEDULER=$2
    local DIRECTION=$3
    local TEMPERATURE=$4
    local CLM_BETA=$5
    local CLM_BETA_SCHEDULER=$6
    local MAIN_PROCESS_PORT=$7

    local KD_DIR=${KD_DIR_BASE}_alpha_${START_ALPHA}_${ALPHA_SCHEDULER}_${DIRECTION}_beta_${CLM_BETA}_${CLM_BETA_SCHEDULER}_t${TEMPERATURE}/
    local LOGDIR=${LOGDIR_BASE}_alpha_${START_ALPHA}_${ALPHA_SCHEDULER}_${DIRECTION}_beta_${CLM_BETA}_${CLM_BETA_SCHEDULER}_t${TEMPERATURE}/

    mkdir -p $KD_DIR
    mkdir -p $LOGDIR

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE accelerate launch --main_process_port $MAIN_PROCESS_PORT kd.py \
        --per_device_train_batch_size $BS --block_size 4096 --seed 42 \
        --pretrained_model_name_or_path $MODEL_KD --epochs $EPOCHS \
        --checkpointing_steps 100000 --tensorboard_log_dir $LOGDIR \
        --mode sft --output_ckpt_dir $KD_DIR \
        --text2sql_data_dir $TEXT2SQL_DATA_DIR \
        --table_num 6 --column_num 10 \
        --lr $LR --warmup_ratio $LR_WARMUP_RATIO \
        --clm_beta $CLM_BETA --clm_beta_scheduler $CLM_BETA_SCHEDULER \
        --kd_loss $KD_LOSS --temperature $TEMPERATURE --mixin_alpha -1 \
        --alpha_scheduler $ALPHA_SCHEDULER --min_alpha $MIN_ALPHA --start_alpha $START_ALPHA \
        --alpha_step_size $ALPHA_STEP_SIZE --alpha_warmup_ratio $ALPHA_WARMUP_RATIO \
        --imp_weight $IMG_WEIGHT \
        --direction $DIRECTION \

    echo "Sleeping for 5 minutes..."
    sleep 300
}

# Running with different start alpha values and ports

# EXP: mixin scheduler (sin)

# start_alpha | alpha_direction | temperature | clm_beta_scheduler | port

# run_training 0.0 cos inverse 2 0.5 const 29548
# run_training 0.1 cos inverse 2 0.5 const 29543
# run_training 0.2 cos inverse 2 0.5 const 25548
# run_training 0.3 cos inverse 2 0.5 const 29544
# run_training 0.4 cos inverse 2 0.5 const 29547
# run_training 0.5 cos inverse 2 0.5 const 29542
# run_training 0.6 cos inverse 2 0.5 const 29541
# run_training 0.7 cos inverse 2 0.5 const 29538
# run_training 0.8 cos inverse 2 0.5 const 29528
# run_training 0.9 cos inverse 2 0.5 const 29533
# run_training 1.0 cos inverse 2 0.5 const 29448

# EXP: temperature (cos)

# run_training 0.2 cos normal 0.5 0.5 const 29548
# run_training 0.2 cos normal 1 0.5 const 29548
# run_training 0.2 cos normal 3 0.5 const 29548

# EXP: beta scheduler (cos)

# run_training 0 const normal 2 0.1 cos 29543
# run_training 0 const normal 2 0.2 cos 25548
# run_training 0 const normal 2 0.3 cos 29544
# run_training 0 const normal 2 0.4 cos 29547
# run_training 0 const normal 2 0.5 cos 29542
# run_training 0 const normal 2 0.6 cos 29541
# run_training 0 const normal 2 0.7 cos 29538
# run_training 0 const normal 2 0.8 cos 29528
# run_training 0 const normal 2 0.9 cos 29533
run_training 0 const normal 2 1.0 cos 29448

# bash ./eval_kd.sh
