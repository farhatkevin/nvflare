from transformers import AutoModelForCausalLM, AutoConfig
import torch
import os
import shutil
from pathlib import Path

# Path to your checkpoint
checkpoint_path = "/workspace/NVFlare/examples/advanced/llm_hf/hf_pretrain/workdir/site-lbv1/pretrain/checkpoint-24"

# Create output directory for the safetensors version
output_path = f"{checkpoint_path}_safetensors"
os.makedirs(output_path, exist_ok=True)

# First, copy all the non-model files to the new directory
for file in os.listdir(checkpoint_path):
    if file != "pytorch_model.bin":
        src = os.path.join(checkpoint_path, file)
        dst = os.path.join(output_path, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {file} to output directory")

# Load the model configuration
config = AutoConfig.from_pretrained(checkpoint_path)

# Load the model from the checkpoint (this loads the pytorch_model.bin file)
print("Loading model from checkpoint...")
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    config=config,
    torch_dtype=torch.float16,  # You can change this to the precision you need
)

# Convert and save the model to safetensors format
print("Converting model to safetensors format...")
model.save_pretrained(output_path, safe_serialization=True)

print(f"Conversion complete! Model saved to {output_path}")

# Verify that the conversion worked
if os.path.exists(os.path.join(output_path, "model.safetensors")):
    print("Success: model.safetensors file was created!")
    # Optionally, you can delete the original pytorch_model.bin to save space
    # os.remove(os.path.join(output_path, "pytorch_model.bin"))
else:
    print("Error: model.safetensors file was not created.")


# from transformers import AutoModelForCausalLM
# import torch

# # Path to your safetensors model
# model_path = "/workspace/NVFlare/examples/advanced/llm_hf/hf_pretrain/workdir/site-lbv1/pretrain/checkpoint-24_safetensors"

# # Load the model
# print("Loading model from safetensors...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,  # Same precision as when you saved it
# )

# # Print model architecture
# print("\nModel Architecture:")
# print(model)

# # Print each named parameter and its shape
# print("\nDetailed Layer Information:")
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Shape: {param.shape}, Size: {param.numel():,} parameters")

# # Count total parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"\nTotal Parameters: {total_params:,}")

# # Optional: Print model configuration details
# print("\nModel Configuration:")
# print(model.config)

# # Check which weights are loaded as safetensors
# print("\nModel Weights Format:")
# import os
# if os.path.exists(os.path.join(model_path, "model.safetensors")):
#     print("Main model weights are in safetensors format (model.safetensors)")
# elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
#     print("Main model weights are in PyTorch format (pytorch_model.bin)")

# # Check for sharded weights
# safetensor_shards = [f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors")]
# if safetensor_shards:
#     print(f"Model weights are sharded into {len(safetensor_shards)} safetensors files:")
#     for shard in sorted(safetensor_shards):
#         print(f"  - {shard}")