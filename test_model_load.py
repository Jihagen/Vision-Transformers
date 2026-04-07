#!/usr/bin/env python
from __future__ import annotations

import os

# Set cache/offload paths BEFORE HF/transformers imports
os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_HUB_CACHE"] = "D:/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/transformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from pathlib import Path

from dotenv import load_dotenv
import torch
from huggingface_hub import login
from huggingface_hub import constants as hf_constants
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

# ---- logging ----
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "test_model_load.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- env / auth ----
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env")

login(token=hf_token, add_to_git_credential=False)

logger.info(f"HF_HOME env: {os.environ.get('HF_HOME')}")
logger.info(f"HF_HUB_CACHE env: {os.environ.get('HF_HUB_CACHE')}")
logger.info(f"TRANSFORMERS_CACHE env: {os.environ.get('TRANSFORMERS_CACHE')}")
logger.info(f"Resolved HF_HOME: {hf_constants.HF_HOME}")
logger.info(f"Resolved HF_HUB_CACHE: {hf_constants.HF_HUB_CACHE}")

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
OFFLOAD_DIR = "D:/offload_dir"
Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

logger.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ---- step 1: config ----
logger.info("Loading config...")
cfg = AutoConfig.from_pretrained(
    MODEL_ID,
    token=hf_token,
    local_files_only=True,
)
logger.info(f"Config loaded: model_type={cfg.model_type}")

# ---- step 2: processor ----
logger.info("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    token=hf_token,
    local_files_only=True,
)
logger.info(f"Processor loaded: {type(processor).__name__}")

# ---- step 3: model load only ----
logger.info("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    token=hf_token,
    local_files_only=True,
    device_map="sequential",
    torch_dtype=torch.float16,   # or float32 if still unstable
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
)
logger.info(f"Model loaded: {model.__class__.__name__}")

if hasattr(model, "hf_device_map"):
    logger.info("Device map:")
    for k, v in model.hf_device_map.items():
        logger.info(f"  {k} -> {v}")

logger.info("SUCCESS: config, processor, and model loaded.")