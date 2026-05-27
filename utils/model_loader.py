"""
Model and processor loading for Llama-4-Scout-17B (or any HF vision-language model).

Handles BitsAndBytes quantization config, device mapping, and CPU offloading
needed to fit the model on the available GPU cluster nodes.

Migrated from: get_activations_base.py (model/processor init block)
"""

from __future__ import annotations

# from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
# import torch


MODEL_PATH_DEFAULT = "/anvme/workspace/iwi5268h-vision-transformers/hf_cache"


def load_model(
    model_path: str = MODEL_PATH_DEFAULT,
    offload_dir: str | None = None,
    quantize: bool = True,
    device_map: str = "auto",
):
    """
    Load the vision-language model with optional 4-bit quantization and CPU offloading.

    Returns:
        model: loaded HuggingFace model in eval mode
    """
    raise NotImplementedError


def load_processor(model_path: str = MODEL_PATH_DEFAULT):
    """
    Load the tokenizer/processor for the model.

    Returns:
        processor: AutoProcessor instance
    """
    raise NotImplementedError


def load_model_and_processor(
    model_path: str = MODEL_PATH_DEFAULT,
    offload_dir: str | None = None,
    quantize: bool = True,
):
    """Convenience wrapper returning (model, processor)."""
    model = load_model(model_path, offload_dir=offload_dir, quantize=quantize)
    processor = load_processor(model_path)
    return model, processor
