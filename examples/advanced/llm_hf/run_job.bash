python3 llm_hf_fl_job_pretrain.py \
  --client_ids code math lbv1 \
  --data_path /data/input/kevinf/nvflare/examples/advanced/llm_hf/data  \
  --workspace_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/workdir \
  --job_dir /data/input/kevinf/nvflare/examples/advanced/llm_hf/hf_pretrain/jobdir \
  --train_mode pretrain \
  --model_name_or_path allenai/OLMo-2-1124-7B \
  --gpu 5,6,7 \
  --quantize_mode float16 \
  --threads 1  > run1.txt 2>&1 \
  --max_tokens 7M \
  # --model_name_or_path allenai/OLMo-2-0425-1B \