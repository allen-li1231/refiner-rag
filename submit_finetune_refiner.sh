#!/usr/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=i64m1tga800ue
#SBATCH -J SFTMonitor
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

#- Log information
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Job starts at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Runs at:"
echo "$(hostnamectl)"
# Number of total processes
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# CUDA and Python environment
module load cuda/12.2
# source ~/miniconda3/bin/activate dl
eval "$(micromamba shell hook --shell bash)"
micromamba activate dl
export CUDA_LAUNCH_BLOCKING=1

# Model parameters
TASK=monitor
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training Llama-2 $MODEL_SIZE model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
./finetune.py \
    --task $TASK \
    --model_name_or_path "$MODEL_NAME" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps $GRADIENT_ACC_STEPS \
    --checkpointing_steps $((GRADIENT_ACC_STEPS * 10)) \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --seed 633 \
    --num_train_epochs 3 \
    --learning_rate 4e-5 \
    --train_file ../train_data/llama3_truncated/arc_c_hotpotqa_triviaqa_truncated.jsonl \
    --output_dir ./checkpoint/monitor_from_llama3_truncated_no_special_token/ \
    --use_flash_attn

echo "Q.E.D"