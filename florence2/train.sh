CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type florence-2-large-ft \
    --num_train_epochs 10 \
    --warmup_ratio 0.4 \
    --batch_size 4 \
    --dataset /home/oem/work/florence2/data.jsonl \
    --lora_target_modules ALL