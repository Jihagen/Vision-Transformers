# test_model_load.py
#!/usr/bin/env python3
import os, sys
import argparse
import torch 
# 1) force your dev/hf_deps and transformers/dev ahead of site-packages:
DEVROOT = "/anvme/workspace/iwi5268h-bigdata/dev"
sys.path.insert(0, os.path.join(DEVROOT, "hf_deps"))
sys.path.insert(0, os.path.join(DEVROOT, "transformers", "src"))

# 2) disable path-as-repo validation so HF will accept a bare directory path
import huggingface_hub.utils._validators as _hf_val
_hf_val.validate_repo_id = lambda *args, **kwargs: None
import transformers.utils.hub as _tf_hub
_tf_hub.validate_repo_id = lambda *args, **kwargs: None

# 3) now import what you need
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()
MODEL_PATH = args.model_path

print("→ Loading config from:", MODEL_PATH)
cfg = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
print("✅ Config loaded; model_type =", cfg.model_type)

print("→ Loading processor from:", MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
print("✅ Processor loaded; processor class =", type(processor).__name__)

OFFLOAD_DIR = "./offload_dir"

print(f"→ Loading model from: {MODEL_PATH} with offload folder: {OFFLOAD_DIR}")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
)
print("✅ Model loaded; architecture =", model.__class__.__name__)