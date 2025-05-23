# python3 llm_hf_fl_job_pretrain.py \
#   --client_ids code math lbv1 \
#   --data_path /data/input/kevinf/nvflare/examples/advanced/llm_hf/data  \
#   --workspace_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/workdir \
#   --job_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/jobdir \
#   --train_mode pretrain \
#   --model_name_or_path allenai/OLMo-2-1124-7B \
#   --gpu 0,1,2 \
#   --quantize_mode float16 \
#   --threads 3 > run1.txt 2>&1 \
#   --max_tokens 100k \
#   # --model_name_or_path allenai/OLMo-2-0425-1B \

#!/bin/bash

# MODEL_NAME="allenai/OLMo-2-0425-1B"
MODEL_NAME="allenai/OLMo-2-1124-7B"
MAX_TOKENS="1M"
GPU_IDS="0,1,2"
THREADS=3

# Extract clean model tag like 'olmo2-7B' from full model path
MODEL_TAG=$(basename "$MODEL_NAME" | sed -E 's/.*OLMo-2-[0-9]+-(.*)/olmo2-\1/I')

NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
RUN_NAME="${MODEL_TAG}_toks${MAX_TOKENS}_gpus${NUM_GPUS}_thr${THREADS}_with_timeout_fix_v2"
LOG_FILE="${RUN_NAME}.log"

echo "Running with timeout fixes for large model transfers..."
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME"

# Execute script
python3 llm_hf_fl_job_pretrain.py \
  --client_ids code math lbv1 \
  --data_path /data/input/kevinf/nvflare/examples/advanced/llm_hf/data \
  --workspace_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/workdir \
  --job_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/jobdir \
  --train_mode pretrain \
  --model_name_or_path "$MODEL_NAME" \
  --gpu "$GPU_IDS" \
  --quantize_mode float16 \
  --threads "$THREADS" \
  --max_tokens "$MAX_TOKENS" \
  > "logs/$LOG_FILE" 2>&1

echo "Job completed! Check logs/$LOG_FILE for results."
