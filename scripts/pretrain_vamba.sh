# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6 libaio-dev -y

export PYTHONPATH=$(pwd)

model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"
max_img_seq_len=120000
max_txt_seq_len=124000

lora_enabled=false
qlora_enabled=false
global_batch_size=16
trainable_modules="cross_attn,vision_mixer,alpha"

if [ -z "$DATA_CONFIG_FILE" ]; then
    DATA_CONFIG_FILE="train/data_configs/data_cc12m_pixelprose_packed.yaml"
fi

OUTPUT_DIR="output/pretraining"
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="vamba_cc12m_pixelprose_packed_lr1e-5"
fi

export WANDB_API_KEY="<YOUR WANDB API KEY>"
export WANDB_PROJECT="vamba_pretraining"
export WANDB_NAME=$RUN_NAME

# set default TRAINING_STEPS and LEARNING_RATE values
if [ -z "$TRAINING_EPOCHS" ]; then
    TRAINING_EPOCHS=1
fi

if [ -z "$LEARNING_RATE" ]; then
    LEARNING_RATE=1e-5
fi

if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    if [ $qlora_enabled = true ]; then
        echo "qlora & dora is enabled"
        RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}_qlora"
    else
        RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}_lora"
    fi
else
    echo "lora is disabled"
    RUN_NAME="${RUN_NAME}_img${max_img_seq_len}_txt${max_txt_seq_len}"
fi
echo "RUN_NAME = $RUN_NAME"


# resume from checkpoint
resume_from_checkpoint=""
if [ -d $resume_from_checkpoint ]; then
    echo "resume_from_checkpoint = $resume_from_checkpoint"
    export WANDB_LAST_RUN_ID="your_last_run_id"
else
    echo "No checkpoint found, training from scratch"
fi

export NCCL_DEBUG=INFO;
export CXX=g++;

MAIN_PROCESS_IP=${MLP_WORKER_0_HOST}
MAIN_PROCESS_PORT=${MLP_WORKER_0_PORT}
NUM_MACHINES=${MLP_WORKER_NUM}
MACHINE_RANK=${MLP_ROLE_INDEX}


NGPU_PER_NODE=${MLP_WORKER_GPU}
NUM_PROCESSES=$((${NUM_MACHINES} * ${NGPU_PER_NODE}))
NUM_WORKERS=16

if [ $NUM_WORKERS -gt 112 ]; then
    NUM_WORKERS=112
fi


echo MAIN_PROCESS_IP=${MAIN_PROCESS_IP}
echo MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}
echo NUM_MACHINES=${NUM_MACHINES}
echo MACHINE_RANK=${MACHINE_RANK}
echo NUM_PROCESSES=${NUM_PROCESSES}
echo NUM_WORKERS=${NUM_WORKERS}
echo "Running ${RUN_NAME}"


if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    config_file="train/accelerate_configs/accelerate_config_zero2.yaml"
    echo $config_file
else
    echo "lora is disabled"
    config_file="train/accelerate_configs/accelerate_config_zero3.yaml"
    echo $config_file
fi

if [ -z "$per_device_train_batch_size" ]; then
    per_device_train_batch_size=1
fi
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $NUM_PROCESSES)))
echo gradient_accumulation_steps=$global_batch_size / \($per_device_train_batch_size \* $NUM_PROCESSES\) = $gradient_accumulation_steps


accelerate launch --config_file=$config_file \
    --machine_rank ${MACHINE_RANK} --main_process_ip ${MAIN_PROCESS_IP} --main_process_port ${MAIN_PROCESS_PORT} \
    --num_machines=${NUM_MACHINES} --num_processes=${NUM_PROCESSES} \
    train/train_vamba.py \
    --model_name_or_path $model_name_or_path \
    --data_config_file $DATA_CONFIG_FILE \
    --run_name $RUN_NAME \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAINING_EPOCHS \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 10 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers $NUM_WORKERS \
    --report_to "wandb" \
    --do_train True \
    --trainable_modules "$trainable_modules" \
    --lora_enabled $lora_enabled \
    --qlora_enabled $qlora_enabled \
    --max_img_seq_len $max_img_seq_len \
    --max_txt_seq_len $max_txt_seq_len \
    --init_cross_attn_weights_from_self_attn \
    --gradient_checkpointing_kwargs '{"use_reentrant":"False"}' \
    --enable_vision_mamba_mixer \
    --use_alpha_balancing \
    --resume_from_checkpoint "$resume_from_checkpoint"