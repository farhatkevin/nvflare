python3 llm_hf_fl_job_pretrain.py \
  --client_ids code math lbv1 \
  --data_path $(pwd)/data \
  --workspace_dir /workspace/hf_pretrain/workdir \
  --job_dir /workspace/hf_pretrain/jobdir \
  --train_mode pretrain \
  --model_name_or_path allenai/OLMo-2-0425-1B \
  --gpu 0 \
  --quantize_mode float16
  # --threads 1 \