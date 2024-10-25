#!/bin/bash
#SBATCH -J longsft
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:2
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-sft-ds-w-ICL-raw-out.log
#SBATCH -e logs/%j-sft-ds-w-ICL-raw-err.log
# The shell provided in the README.

export WANDB_API_KEY="7cb3dada5935174b3d1b35a051f0e5cabc2d7be1"
DS_SKIP_CUDA_CHECK=1 torchrun --nproc_per_node=2 --master_port=$((1024 + RANDOM % (65535 - 1024 + 1))) train.py \
    --report_to wandb \
    --run_name debug \
    --model_name_or_path ./models/Meta-Llama-3-8B-Instruct \
    --data_paths data/longbook-choice-eng-train-sent.json data/longbook-qa-eng-train-sent.json data/longbook-sum-eng-train-sent.json data/longdialogue-qa-eng-train-sent.json data/quality-train-sent.json \
    --len_segment 8 \
    --len_offset 3 \
    --block_size 256 \
    --model_max_length 8000 \
    --bf16 True \
    --output_dir models/sft-w-ICL-raw \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5 \
    --save_only_model True \
    --logging_strategy steps \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True \
    --enable_lora True \
    --lora_rank 8 \
    --load_in_4bit True
