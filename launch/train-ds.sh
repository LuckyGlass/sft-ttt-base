# The shell provided in the README.

torchrun --nproc_per_node=4 --master_port=$RANDOM train.py \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --data_path data/timeline-dev.json \
    --len_segments 8 \
    --len_offsets 3 \
    --block_size 256 \
    --model_max_length 8000 \
    --bf16 True \
    --output_dir models/sft-w-ICL \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True \
    --enable_lora True \
    --lora_rank 8 \
    --load_in_4bit True
