python3 llm_hf_fl_job_pretrain.py \
  --client_ids code math lbv1 \
  --data_path "/workspace/NVFlare/examples/advanced/llm_hf/data" \
  --workspace_dir /workspace/NVFlare/examples/advanced/llm_hf/hf_pretrain/workdir \
  --job_dir /workspace/NVFlare/examples/advanced/llm_hf/hf_pretrain/jobdir \
  --train_mode pretrain \
  --model_name_or_path allenai/OLMo-2-0425-1B \
  --gpu 0,1 \
  --quantize_mode float16 \
  --threads 1  > run1.txt 2>&1