# from transformers import AutoModelForCausalLM, AutoConfig
# import torch
# import os
# from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
# from nvflare.apis.fl_context import FLContext
# import time

# class HFFileModelPersistor(PTFileModelPersistor):
#     def __init__(
#         self,
#         exclude_vars=None,
#         model=None,
#         global_model_file_name="global_model",  # No .pt extension since we're saving a directory
#         best_global_model_file_name="best_global_model", 
#         source_ckpt_file_full_name=None,
#         filter_id=None,
#         load_weights_only=False,
#         model_name_or_path=None,  # Model name or path for AutoModelForCausalLM
#         allow_numpy_conversion=True,
#     ):
#         super().__init__(
#             exclude_vars=exclude_vars,
#             model=model,
#             global_model_file_name=global_model_file_name,
#             best_global_model_file_name=best_global_model_file_name,
#             source_ckpt_file_full_name=source_ckpt_file_full_name,
#             filter_id=filter_id,
#             load_weights_only=load_weights_only,
#             allow_numpy_conversion=allow_numpy_conversion,
#         )
#         self.model_name_or_path = model_name_or_path
        
#     def save_model_file(self, save_path: str):
#         """Override to save in HF format using save_pretrained"""
#         print(f"⭐ Saving global model to {save_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
#         # Get model weights
#         save_dict = self.persistence_manager.to_persistence_dict()
        
#         # Extract model state dict (it might be nested under 'model' key)
#         if isinstance(save_dict, dict) and "model" in save_dict:
#             state_dict = save_dict["model"]
#         else:
#             state_dict = save_dict
            
#         # Log model info
#         model_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
#         print(f"  • Model has {len(state_dict)} layers and {model_params:,} parameters")
        
#         try:
#             # First save traditional format as backup (for compatibility)
#             torch.save(save_dict, save_path + ".pt")  # Add .pt extension
#             print(f"  ✓ Saved PyTorch backup to {save_path}.pt")
            
#             # Remove .pt extension if it exists in the path
#             if save_path.endswith('.pt'):
#                 hf_save_path = save_path[:-3]
#             else:
#                 hf_save_path = save_path
                
#             # Create models with proper architecture from our model name/path
#             if self.model_name_or_path:
#                 # Initialize model with the right architecture
#                 config = AutoConfig.from_pretrained(self.model_name_or_path)
#                 model = AutoModelForCausalLM.from_pretrained(
#                     self.model_name_or_path,
#                     config=config,
#                     torch_dtype=torch.float16  # Use appropriate dtype
#                 )
                
#                 # Adapt state dict keys if needed
#                 adapted_state_dict = {}
#                 for key, value in state_dict.items():
#                     # Handle keys that might have 'model.' prefix
#                     if key.startswith("model."):
#                         adapted_state_dict[key[6:]] = value
#                     else:
#                         adapted_state_dict[key] = value
                
#                 # Load adapted state dict into model
#                 model.load_state_dict(adapted_state_dict)
                
#                 # Save using HF's save_pretrained
#                 model.save_pretrained(hf_save_path, safe_serialization=True)
#                 print(f"  ✓ Saved HF model to {hf_save_path} using save_pretrained")
#             else:
#                 print(f"  ⚠ Could not save in HF format - no model_name_or_path provided")
#         except Exception as e:
#             print(f"  ❌ Error saving model in HF format: {e}")

from transformers import AutoModelForCausalLM, AutoConfig
import torch
import os
import time
import traceback
import numpy as np

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.model import ModelLearnable

class HFFileModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        exclude_vars=None,
        model=None,
        global_model_file_name="global_model",
        best_global_model_file_name="best_global_model", 
        source_ckpt_file_full_name=None,
        filter_id=None,
        load_weights_only=False,
        model_name_or_path="meta-llama/llama-3.2-1b",  # Specify your model type here
        allow_numpy_conversion=True,
    ):
        super().__init__(
            exclude_vars=exclude_vars,
            model=model,
            global_model_file_name=global_model_file_name,
            best_global_model_file_name=best_global_model_file_name,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
            filter_id=filter_id,
            load_weights_only=load_weights_only,
            allow_numpy_conversion=allow_numpy_conversion,
        )
        self.model_name_or_path = model_name_or_path
    

    # def save_model_file(self, save_path: str):
    #     """Save model in both PyTorch and HF formats with verbose debugging"""
    #     print(f"💾 SAVING MODEL FILE: {save_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    #     # Get model state dict
    #     save_dict = self.persistence_manager.to_persistence_dict()
        
    #     # First save in original format as backup
    #     torch.save(save_dict, save_path)
    #     print(f"✅ SAVED PyTorch model to {save_path}")
        
    #     try:
    #         # Remove .pt extension if it exists in the path
    #         if save_path.endswith('.pt'):
    #             hf_save_path = save_path[:-3]
    #         else:
    #             hf_save_path = save_path

    #         hf_save_path = hf_save_path + "_hf"
                
    #         print(f"🔄 Attempting to save in HF format to {hf_save_path}")
            
    #         # Get the state dict
    #         if isinstance(save_dict, dict) and "model" in save_dict:
    #             state_dict = save_dict["model"]
    #             print(f"  ℹ️ Found 'model' key in state_dict")
    #         else:
    #             state_dict = save_dict
    #             print(f"  ℹ️ Using entire state_dict as model weights")
            
    #         # Debug info for model architecture
    #         print(f"  📊 State dict has {len(state_dict)} keys")
    #         sample_keys = list(state_dict.keys())[:3]
    #         print(f"  🔑 Sample keys: {sample_keys}")
            
    #         # Create a HuggingFace model to save
    #         print(f"  🔄 Loading model architecture from {self.model_name_or_path}")
    #         config = AutoConfig.from_pretrained(self.model_name_or_path)
    #         model = AutoModelForCausalLM.from_pretrained(
    #             self.model_name_or_path,
    #             config=config,
    #             torch_dtype=torch.float16
    #         )
            
    #         # Convert numpy arrays to tensors and adapt state_dict keys
    #         print(f"  🔄 Converting numpy arrays to tensors and adapting keys")
    #         adapted_state_dict = {}
    #         for key, value in state_dict.items():
    #             # Convert NumPy arrays to tensors
    #             if isinstance(value, np.ndarray):
    #                 tensor_value = torch.tensor(value)
    #             elif isinstance(value, torch.Tensor):
    #                 tensor_value = value
    #             else:
    #                 print(f"    ⚠️ Skipping non-tensor value: {key}, type: {type(value)}")
    #                 continue
                    
    #             # Handle keys with 'model.' prefix
    #             if key.startswith("model.model."):
    #                 adapted_key = key[6:]  # Remove first 'model.'
    #                 adapted_state_dict[adapted_key] = tensor_value
    #                 print(f"    🔁 Renamed key: {key} -> {adapted_key}")
    #             elif key.startswith("model."):
    #                 adapted_state_dict[key] = tensor_value
    #             else:
    #                 adapted_state_dict[key] = tensor_value
            
    #         # Load state dict into model
    #         print(f"  🔄 Loading state dict into model")
    #         try:
    #             missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
    #             print(f"  ✅ Loaded state dict with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
    #             if missing_keys:
    #                 print(f"    ⚠️ First few missing keys: {missing_keys[:5]}")
    #             if unexpected_keys:
    #                 print(f"    ⚠️ First few unexpected keys: {unexpected_keys[:5]}")
    #         except Exception as e:
    #             print(f"  ❌ Error loading state dict: {e}")
    #             print(f"  ⚠️ Will create empty model save instead")
                
    #         # Save using HF's save_pretrained
    #         print(f"  🔄 Saving with model.save_pretrained({hf_save_path}, safe_serialization=True)")
    #         os.makedirs(hf_save_path, exist_ok=True)
    #         model.save_pretrained(hf_save_path, safe_serialization=True)
    #         print(f"✅ SAVED HF model to {hf_save_path}")
            
    #         # Verify the HF directory contents
    #         if os.path.exists(hf_save_path):
    #             files = os.listdir(hf_save_path)
    #             print(f"  📁 HF directory contents: {files}")
    #         else:
    #             print(f"  ❌ HF directory {hf_save_path} does not exist after save!")
                
    #     except Exception as e:
    #         print(f"❌ ERROR saving model in HF format: {e}")
    #         print(f"Traceback: {traceback.format_exc()}")
    #         print(f"Model was saved in PyTorch format only at {save_path}")

    # def save_model_file(self, save_path: str):
    #     """Save model in both PyTorch and HF formats with verbose debugging"""
    #     print(f"💾 SAVING MODEL FILE: {save_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    #     # Get model state dict
    #     save_dict = self.persistence_manager.to_persistence_dict()
        
    #     # First save in original format as backup
    #     torch.save(save_dict, f"{save_path}.pt")  # Add .pt extension
    #     print(f"✅ SAVED PyTorch model to {save_path}.pt")
        
    #     try:
    #         # Create a new directory path for HF model
    #         hf_save_path = save_path + "_hf"
    #         print(f"🔄 Attempting to save in HF format to {hf_save_path}")
            
    #         # Get the state dict
    #         if isinstance(save_dict, dict) and "model" in save_dict:
    #             state_dict = save_dict["model"]
    #             print(f"  ℹ️ Found 'model' key in state_dict")
    #         else:
    #             state_dict = save_dict
    #             print(f"  ℹ️ Using entire state_dict as model weights")
            
    #         # Debug info for model architecture
    #         print(f"  📊 State dict has {len(state_dict)} keys")
            
    #         # Create a HuggingFace model to save
    #         print(f"  🔄 Loading model and tokenizer from {self.model_name_or_path}")
    #         from transformers import AutoTokenizer
            
    #         # Load model configuration and create model
    #         config = AutoConfig.from_pretrained(self.model_name_or_path)
    #         model = AutoModelForCausalLM.from_pretrained(
    #             self.model_name_or_path,
    #             config=config,
    #             torch_dtype=torch.float16
    #         )
            
    #         # Load tokenizer
    #         tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            
    #         # Convert numpy arrays to tensors and adapt state_dict keys
    #         print(f"  🔄 Converting numpy arrays to tensors and adapting keys")
    #         adapted_state_dict = {}
    #         for key, value in state_dict.items():
    #             # Convert NumPy arrays to tensors
    #             if isinstance(value, np.ndarray):
    #                 tensor_value = torch.tensor(value)
    #             elif isinstance(value, torch.Tensor):
    #                 tensor_value = value
    #             else:
    #                 print(f"    ⚠️ Skipping non-tensor value: {key}, type: {type(value)}")
    #                 continue
                    
    #             # Handle keys with 'model.' prefix
    #             if key.startswith("model.model."):
    #                 adapted_key = key[6:]  # Remove first 'model.'
    #                 adapted_state_dict[adapted_key] = tensor_value
    #             elif key.startswith("model."):
    #                 adapted_state_dict[key] = tensor_value
    #             else:
    #                 adapted_state_dict[key] = tensor_value
            
    #         # Load state dict into model
    #         print(f"  🔄 Loading state dict into model")
    #         try:
    #             missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
    #             print(f"  ✅ Loaded state dict with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
    #         except Exception as e:
    #             print(f"  ⚠️ Warning loading state dict: {e}")
    #             print(f"  🔄 Continuing with original model weights")
            
    #         # Make sure directory doesn't exist before creating
    #         if os.path.exists(hf_save_path) and os.path.isfile(hf_save_path):
    #             print(f"  ⚠️ Removing existing file at {hf_save_path}")
    #             os.remove(hf_save_path)
            
    #         # Create directory
    #         os.makedirs(hf_save_path, exist_ok=True)
            
    #         # Save model and tokenizer
    #         print(f"  🔄 Saving model to {hf_save_path}")
    #         model.save_pretrained(hf_save_path, safe_serialization=True)
            
    #         print(f"  🔄 Saving tokenizer to {hf_save_path}")
    #         tokenizer.save_pretrained(hf_save_path)
            
    #         print(f"✅ SAVED complete HF model checkpoint to {hf_save_path}")
            
    #         # Verify the HF directory contents
    #         files = os.listdir(hf_save_path)
    #         print(f"  📁 HF directory contents: {files}")
                
    #     except Exception as e:
    #         print(f"❌ ERROR saving model in HF format: {e}")
    #         print(f"Traceback: {traceback.format_exc()}")
    #         print(f"Model was saved in PyTorch format only at {save_path}")

    def save_model_file(self, save_path: str):
        """Optimized version with faster HF operations"""
        import time
        agg_start = time.time()
        print(f"💾 SAVING MODEL FILE: {save_path}")
        
        # Get model state dict
        save_dict = self.persistence_manager.to_persistence_dict()
        
        # First save in original format
        torch.save(save_dict, save_path + ".pt")
        print(f"✅ SAVED PyTorch model to {save_path}.pt")
        
        try:
            # Create HF save path
            hf_save_path = save_path + "_hf"
            print(f"🔄 Saving in HF format to {hf_save_path}")
            
            # Get the state dict
            if isinstance(save_dict, dict) and "model" in save_dict:
                state_dict = save_dict["model"]
            else:
                state_dict = save_dict
                
            # Create directory
            os.makedirs(hf_save_path, exist_ok=True)
            
            # Save config.json directly
            config_path = os.path.join(hf_save_path, "config.json")
            if not os.path.exists(config_path) and hasattr(self, "model_name_or_path"):
                try:
                    # Faster than loading the entire model
                    from huggingface_hub import hf_hub_download
                    import shutil
                    
                    # Download just the config file
                    config_file = hf_hub_download(
                        repo_id=self.model_name_or_path,
                        filename="config.json"
                    )
                    shutil.copy(config_file, config_path)
                    print(f"  ✅ Copied config.json")
                    
                    # Also get tokenizer files (minimal set)
                    for filename in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
                        try:
                            file_path = hf_hub_download(
                                repo_id=self.model_name_or_path,
                                filename=filename
                            )
                            shutil.copy(file_path, os.path.join(hf_save_path, filename))
                            print(f"  ✅ Copied {filename}")
                        except Exception as e:
                            print(f"  ⚠️ Could not copy {filename}: {e}")
                except Exception as e:
                    print(f"  ⚠️ Could not download config: {e}")
            
            # Save model weights directly in safetensors format
            print(f"  🔄 Saving model weights")
            
            # Convert to proper format and save
            from safetensors.torch import save_file
            
            # Make sure all weights are tensors, not numpy arrays
            tensor_dict = {}
            for key, value in state_dict.items():
                # Clean key name (remove model. prefix if present)
                clean_key = key
                if key.startswith("model.model."):
                    clean_key = key[6:]  # Remove first 'model.'
                elif key.startswith("model."):
                    clean_key = key
                    
                # Convert to tensor if numpy
                if isinstance(value, np.ndarray):
                    tensor_dict[clean_key] = torch.tensor(value)
                elif isinstance(value, torch.Tensor):
                    tensor_dict[clean_key] = value
            
            # Save weights directly using safetensors
            safetensors_path = os.path.join(hf_save_path, "model.safetensors")
            save_file(tensor_dict, safetensors_path)
            print(f"  ✅ Saved model weights to {safetensors_path}")
            
            # Verify the HF directory contents
            files = os.listdir(hf_save_path)
            print(f"  📁 HF directory contents: {files}")
                
        except Exception as e:
            print(f"❌ ERROR saving model in HF format: {e}")
            print(f"Model was saved in PyTorch format only at {save_path}.pt")
        agg_end = time.time()
        elapsed = agg_end - agg_start
        print(f"[AGGREGATION THROUGHPUT] Model save/aggregation time: {elapsed:.2f} seconds")
        # Optionally, save to a log file
        agg_log_file = save_path + "_aggregation_time.txt"
        with open(agg_log_file, "w") as f:
            f.write(f"Aggregation/save time: {elapsed:.2f} seconds\n")

    def handle_event(self, event: str, fl_ctx: FLContext):
        """Override to add debugging for events"""
        print(f"🔔 EVENT RECEIVED: {event}")
        
        if event == EventType.START_RUN:
            print(f"🚀 Initializing persistor")
            self._initialize(fl_ctx)
            
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            print(f"🏆 BEST MODEL EVENT: Saving best global model")
            # save the current model as the best model, or the global best model if available
            ml = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            if ml:
                print(f"  ✅ Got global model from context")
                self._get_persistence_manager(fl_ctx).update(ml)
            else:
                print(f"  ⚠️ No global model found in context")
                
            print(f"  💾 Saving to {self._best_ckpt_save_path}")
            self.save_model_file(self._best_ckpt_save_path)
            print(f"  ✅ Best model saved!")
    
    # def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
    #     """Override to add more debugging"""
    #     print(f"📥 SAVE MODEL called at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    #     self._get_persistence_manager(fl_ctx).update(ml)
    #     print(f"  💾 Saving to {self._ckpt_save_path}")
    #     self.save_model_file(self._ckpt_save_path)
        
    #     # List all files in the directory to verify
    #     log_dir = os.path.dirname(self._ckpt_save_path)
    #     print(f"  📂 Contents of {log_dir}:")
    #     for item in os.listdir(log_dir):
    #         itempath = os.path.join(log_dir, item)
    #         if os.path.isdir(itempath):
    #             print(f"    📁 DIR: {item} (contains {len(os.listdir(itempath))} items)")
    #         else:
    #             print(f"    📄 FILE: {item} ({os.path.getsize(itempath)} bytes)")

    # def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
    #     """Override to save per-round checkpoints and log metrics"""

    #     import json 

    #     print(f"📥 SAVE MODEL called at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    #     self._get_persistence_manager(fl_ctx).update(ml)
        
    #     # Get current round from context
    #     current_round = fl_ctx.get_prop("CURRENT_ROUND", 0)
    #     print(f"  ℹ️ Current round: {current_round}")
        
    #     # Save standard checkpoint
    #     print(f"  💾 Saving standard checkpoint to {self._ckpt_save_path}")
    #     self.save_model_file(self._ckpt_save_path)
        
    #     # Also save round-specific checkpoint
    #     round_save_path = f"{self._ckpt_save_path}_round_{current_round}"
    #     print(f"  💾 Saving round-specific checkpoint to {round_save_path}")
    #     self.save_model_file(round_save_path)
        
    #     # List all files in the directory to verify
    #     log_dir = os.path.dirname(self._ckpt_save_path)
    #     print(f"  📂 Contents of {log_dir}:")
    #     for item in os.listdir(log_dir):
    #         itempath = os.path.join(log_dir, item)
    #         if os.path.isdir(itempath):
    #             print(f"    📁 DIR: {item} (contains {len(os.listdir(itempath))} items)")
    #         else:
    #             print(f"    📄 FILE: {item} ({os.path.getsize(itempath)} bytes)")

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        """Save model with round information from meta"""
        import time
        agg_start = time.time()
        print(f"📥 SAVE MODEL called")
        self._get_persistence_manager(fl_ctx).update(ml)
        
        # Try to get round information from model meta
        current_round = None

        if hasattr(ml, "meta") and ml.meta:
            if "CURRENT_ROUND" in ml.meta:
                current_round = ml.meta["CURRENT_ROUND"]
            elif "current_round" in ml.meta:
                current_round = ml.meta["current_round"]
        
        # Add debug to see all available meta keys
        if hasattr(ml, "meta") and ml.meta:
            print(f"  🔍 Available meta keys: {list(ml.meta.keys() if ml.meta else [])}")

        if ml["meta"] is not None:
            print(f"  🔍 Model meta: {ml['meta']}")
            if "CURRENT_ROUND" in ml["meta"]:
                current_round = ml["meta"]["CURRENT_ROUND"]
            elif "current_round" in ml["meta"]:
                current_round = ml["meta"]["current_round"]
            print(f"  🔍 Found current round: {current_round}")
        
        # If we found round information
        if current_round is not None:
            print(f"  ℹ️ Current round: {current_round}")
            
            # Save both standard and round-specific checkpoints
            # self.save_model_file(self._ckpt_save_path)  # Standard
            round_path = f"{os.path.dirname(self._ckpt_save_path)}/round_{current_round}_model"
            self.save_model_file(round_path)  # Round-specific
        else:
            print(f"  ⚠️ No round information found in meta")
            self.save_model_file(self._ckpt_save_path)  # Save standard checkpoint only
        agg_end = time.time()
        elapsed = agg_end - agg_start
        print(f"[AGGREGATION THROUGHPUT] Aggregation+save time: {elapsed:.2f} seconds")
        # Optionally, save to a log file
        agg_log_file = os.path.join(os.path.dirname(self._ckpt_save_path), "aggregation_time.txt")
        with open(agg_log_file, "a") as f:
            f.write(f"Round: {current_round}, Aggregation+save time: {elapsed:.2f} seconds\n")