#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()
MODEL_PATH = args.model_path

OFFLOAD_DIR = "./offload_dir"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("→ Loading config from:", MODEL_PATH)
cfg = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
print("✅ Config loaded; model_type =", cfg.model_type)

print("→ Loading processor from:", MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
print("✅ Processor loaded; processor class =", type(processor).__name__)

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