#!/bin/bash
#SBATCH -J longsft
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-sft-ds-w-ICL-out.log
#SBATCH -e logs/%j-sft-ds-w-ICL-err.log
# The shell provided in the README.

DS_SKIP_CUDA_CHECK=1 torchrun --nproc_per_node=4 --master_port=$RANDOM train.py \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --data_path data/timeline-train.json \
    --len_segment 8 \
    --len_offset 3 \
    --block_size 256 \
    --model_max_length 8000 \
    --bf16 True \
    --output_dir models/sft-w-ICL-raw \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_only_model True \
    --logging_strategy steps \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True \
    --enable_lora True \
    --lora_rank 8 \
    --load_in_4bit True
