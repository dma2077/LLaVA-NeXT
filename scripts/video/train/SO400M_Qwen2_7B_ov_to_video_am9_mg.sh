#!/bin/bash

# Set up the data folder
IMAGE_FOLDER="/map-vepfs/datasets/LLaVA-OneVision-Data-Images"
VIDEO_FOLDER="/llm_reco/dehua/data/LLaVA-Video-178K"
DATA_YAML="/llm_reco/dehua/code/LLaVA-NeXT/scripts/video/train/exp_test.yaml" # e.g exp.yaml

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
# alias python=python3
############### Show Envs ####################
nvidia-smi
source /llm_reco/dehua/anaconda3/bin/activate llava-next
################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
#

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9"
PREV_STAGE_CHECKPOINT="/llm_reco/dehua/model/llava-onevision-qwen2-7b-si"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
# ARNOLD_WORKER_NUM=${1:-1}
# ARNOLD_ID=${2:-0}
# ARNOLD_WORKER_GPU=${3:-8}
# METIS_WORKER_0_HOST=${METIS_WORKER_0_HOST:-127.0.0.1}
export MASTER_ADDR=10.48.48.83

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}
# # 为额外三个变量设置默认值
# if command -v ip > /dev/null 2>&1; then
#     METIS_WORKER_0_HOST=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
# elif command -v hostname > /dev/null 2>&1; then
#     METIS_WORKER_0_HOST=$(hostname -I | awk '{print $1}')
# elif command -v ifconfig > /dev/null 2>&1; then
#     METIS_WORKER_0_HOST=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
# else
#     echo "没有可用的命令来获取IP，请手动设置METIS_WORKER_0_HOST"
#     exit 1
# fi

echo "total workers: ${NNODES}"
echo "cur worker id: ${NODE_RANK}"
echo "gpus per worker: ${GPUS_PER_NODE}"
echo "master ip: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

export HF_HOME=/llm_reco/dehua/model

# ACCELERATE_CPU_AFFINITY=1 
torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;