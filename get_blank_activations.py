#!/usr/bin/env python
from __future__ import annotations
import os, logging, argparse, json, re, gc, time

# ─── CUDA Memory Debugging ──────────────────────────────────────────────────────

os.environ["HF_HOME"] = r"D:\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"
os.environ["HF_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

# ─── Logging ─────────────────────────────────────────────────────────────────────
import logging


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a"),   # save to file
        logging.StreamHandler()                    # still print to terminal
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    logger.info("Logged into Hugging Face via .env HF_TOKEN")
else:
    logger.warning("HF_TOKEN not found in .env or environment")

from huggingface_hub import constants as hf_constants

logger.info(f"HF_HOME env: {os.environ.get('HF_HOME')}")
logger.info(f"HF_HUB_CACHE env: {os.environ.get('HF_HUB_CACHE')}")
logger.info(f"TRANSFORMERS_CACHE env: {os.environ.get('TRANSFORMERS_CACHE')}")
logger.info(f"Resolved HF_HOME: {hf_constants.HF_HOME}")
logger.info(f"Resolved HF_HUB_CACHE: {hf_constants.HF_HUB_CACHE}")

def print_cuda_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved()  / 2**30
        logger.debug(f"{prefix} CUDA mem → Allocated: {a:.2f} GiB, Reserved: {r:.2f} GiB")

def emergency_cleanup(preserve_embeddings=False):
    """Nuclear option for memory cleanup"""
    global layer_embeddings, hook_handles
    if not preserve_embeddings:
        layer_embeddings.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    time.sleep(1)



# ─── Argument Parser ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

# ─── Paths & Checkpoint ──────────────────────────────────────────────────────────
CKPT_PATH = "data/results_blank_activations.npy"
CKPT_TEMP = "data/results_blank_activations_temp.npy"
OFFLOAD_DIR = r"D:\offload_dir"
LABELS_DF_PATH = "data/selected_uniform_total_500.pkl"
os.makedirs("data", exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

MAX_IMAGES = 500
CHECKPOINT_EVERY = 10

IMAGES_ROOT = Path("data/imagesDemographics")



# ─── Load model & processors ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    token=hf_token,
    local_files_only=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
)


logger.info("▶️ device map:")
if hasattr(model, "hf_device_map"):
    logger.info("▶️ device map:")
    for k, v in model.hf_device_map.items():
        logger.info(f"  {k} → {v}")
print_cuda_memory("▶️ initial")

processor = AutoProcessor.from_pretrained(args.model_path, token=hf_token, local_files_only=False)

# ─── Phased capture controls ─────────────────────────────────────────────────────
# We store only when the current phase matches what the hook expects.
CAPTURE_PHASE = "idle"        # "vision_once" | "rating_step" | "idle"

def select_cls(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D] -> CLS token [B, D]
    return x[:, 0, :]

def select_last_token(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D] -> last token [B, D] (the token being emitted in decode)
    return x[:, -1, :]

# ─── Enhanced Model Inspection ───────────────────────────────────────────────────
def inspect_model_structure():
    """Inspect the actual model structure to find hookable layers"""
    logger.info("🔍 Inspecting model structure...")

    # Language layers
    llm_layers = None
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'layers'):
            llm_layers = model.language_model.model.layers
            logger.info(f"Found {len(llm_layers)} language layers")
            for i, layer in enumerate(llm_layers[:3]):  # first 3 preview
                dev = next(layer.parameters()).device if list(layer.parameters()) else 'no params'
                logger.info(f"  Language layer {i}: {type(layer)} on device {dev}")

    # Vision layers (robust path probing)
    vision_paths = [
        ('vision_model.model.layers', lambda: model.vision_model.model.layers if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'model') and hasattr(model.vision_model.model, 'layers') else None),
        ('vision_model.encoder.layers', lambda: model.vision_model.encoder.layers if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'encoder') and hasattr(model.vision_model.encoder, 'layers') else None),
        ('vision_model.vision_model.encoder.layers', lambda: model.vision_model.vision_model.encoder.layers if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'vision_model') and hasattr(model.vision_model.vision_model, 'encoder') and hasattr(model.vision_model.vision_model.encoder, 'layers') else None),
    ]
    vision_layers = None
    vision_path_used = None
    for path_name, path_func in vision_paths:
        try:
            layers = path_func()
            if layers is not None:
                logger.info(f"Found {len(layers)} vision layers at {path_name}")
                vision_layers = layers
                vision_path_used = path_name
                for i, layer in enumerate(layers[:3]):
                    dev = next(layer.parameters()).device if list(layer.parameters()) else 'no params'
                    logger.info(f"  Vision layer {i}: {type(layer)} on device {dev}")
                break
        except Exception as e:
            logger.debug(f"Path {path_name} failed: {e}")

    if vision_layers is None and hasattr(model, 'vision_model'):
        logger.warning("⚠️ Could not find vision layers - checking vision_model structure")
        for attr in dir(model.vision_model):
            if not attr.startswith('_'):
                logger.info(f"  vision_model.{attr}: {type(getattr(model.vision_model, attr))}")

    return llm_layers, vision_layers, vision_path_used

# ─── Hooks (phase-gated + selective) ─────────────────────────────────────────────
layer_embeddings = {}
hook_handles = []
hook_call_count = {}

def get_hook(name: str, store_dict: dict,
             selector: Optional[callable] = None,
             phase_needed: Optional[str] = None,
             dtype: torch.dtype = torch.float16):
    def hook(module, _, output):
        global hook_call_count, CAPTURE_PHASE
        if phase_needed and CAPTURE_PHASE != phase_needed:
            return
        hook_call_count[name] = hook_call_count.get(name, 0) + 1
        try:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out is None or not hasattr(out, 'detach'):
                return
            x = out.detach()
            if selector is not None:
                x = selector(x)
            store_dict[name] = x.to(dtype).cpu()
        except Exception as e:
            logger.warning(f"⚠️ Hook failed on {name}: {e}")
    return hook

# Inspect and register hooks
llm_layers, vision_layers, vision_path = inspect_model_structure()

# Register language model hooks: ALL layers, but only record during rating_step
if llm_layers is not None:
    for i, layer in enumerate(llm_layers):
        handle = layer.register_forward_hook(
            get_hook(
                name=f"llm_layer_{i}_rating_token",
                store_dict=layer_embeddings,
                selector=select_last_token,       # capture the token being emitted in that decode step
                phase_needed="rating_step",       # only during the one-step teacher forcing
                dtype=torch.float16
            )
        )
        hook_handles.append(handle)
    logger.info(f"Registered {len(llm_layers)} LLM hooks (phase=rating_step)")

# Register vision model hooks: ALL layers, CLS only, during the normal pass
if vision_layers is not None:
    for i, layer in enumerate(vision_layers):
        handle = layer.register_forward_hook(
            get_hook(
                name=f"vision_layer_{i}_cls",
                store_dict=layer_embeddings,
                selector=select_cls,              # CLS token only
                phase_needed="vision_once",       # only when we set this phase around generate()
                dtype=torch.float16
            )
        )
        hook_handles.append(handle)
    logger.info(f"Registered {len(vision_layers)} vision hooks using path: {vision_path}")
else:
    logger.warning("⚠️ No vision layers found for hooking")

logger.info(f"Total hooks registered: {len(hook_handles)}")

def remove_batch_dimension(t):
    """Remove batch dimension from tensor and convert to numpy"""
    try:
        cpu_tensor = t.to(torch.float32).cpu()
        if cpu_tensor.dim() > 0:
            if cpu_tensor.shape[0] == 1:
                unbatched = cpu_tensor.squeeze(0)
            else:
                unbatched = cpu_tensor[0]
        else:
            unbatched = cpu_tensor
        return unbatched.numpy()
    except Exception as e:
        logger.warning(f"Error in remove_batch_dimension: {e}, tensor shape: {t.shape}")
        return t.to(torch.float32).cpu().numpy()


def preprocess_image(image_path, max_size=800):
    """Resize image to reduce memory usage"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    return image


# ─── Core Response Function with Phase-Gated Hooks ───────────────────────────────
def extract_last_json_block(text: str) -> dict:
    # Try from the end to avoid "Extra data"
    candidates = re.findall(r'\{.*?\}', text, flags=re.S)
    for raw_json in reversed(candidates):
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found")

def model_response(prompt, image_path, max_length=1000):
    global hook_call_count, CAPTURE_PHASE

    # 0) clear GPU scratch and reset counters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hook_call_count.clear()

    # 1) open & prepare
    image = preprocess_image(image_path)
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=chat_input,
        images=image,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=512,
    )

    """# 2) Move inputs to correct device BEFORE filtering
    target_device = next(model.parameters()).device
    inputs = move_to_device(inputs, target_device)"""

    # 3) Log what we're working with
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            logger.debug(f"Input: {k} {tuple(v.shape)} {v.dtype} @ {v.device}")

    # 4) Filter inputs
    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    extras = set(inputs) - whitelist
    if extras:
        logger.debug(f"Dropping unused inputs: {extras}")
    model_inputs = {k: inputs[k] for k in whitelist if k in inputs}

    # 5) Generate (VISION capture only during this call)
    generation_kwargs = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "use_cache": True,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    logger.info("🚀 Starting generation (vision capture on)...")
    CAPTURE_PHASE = "vision_once"   # only vision hooks will store during this pass
    try:
        print_cuda_memory("Before generate")
        with torch.no_grad():
            gen_ids = model.generate(**model_inputs, **generation_kwargs)
        print_cuda_memory("After generate")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("⚠️ OOM – retrying with 128 tokens")
            emergency_cleanup()
            hook_call_count.clear()
            generation_kwargs["max_new_tokens"] = 128
            with torch.no_grad():
                gen_ids = model.generate(**model_inputs, **generation_kwargs)
        else:
            raise
    finally:
        CAPTURE_PHASE = "idle"

    # 6) Decode + parse JSON once (robust: take the *last* JSON block)
    decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    try:
        data = extract_last_json_block(decoded)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}; raw tail: {decoded[-400:]}")
        raise ValueError("Invalid JSON in response")

    # 7) Teacher-force exactly before the label; capture one token with LLM hooks
    full_seq_ids = gen_ids[0].tolist()
    input_len = model_inputs["input_ids"].shape[1]

    forced_anchor = '{"interestingness":"'
    forced_ids = processor.tokenizer.encode(forced_anchor, add_special_tokens=False)
    prefix_ids = full_seq_ids[:input_len] + forced_ids

    local = {
        "input_ids": torch.tensor([prefix_ids], device=model_inputs["input_ids"].device),
        "attention_mask": torch.ones(
            1, len(prefix_ids),
            device=model_inputs["attention_mask"].device,
            dtype=model_inputs["attention_mask"].dtype
        ),
        "pixel_values": model_inputs["pixel_values"],  # keep vision path active
    }

    logger.info("🎯 Teacher-forced one-step to capture first label token (LLM layers)…")
    CAPTURE_PHASE = "rating_step"
    with torch.no_grad():
        _ = model.generate(
            **local,
            max_new_tokens=1,          # exactly one token → the first label token
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    CAPTURE_PHASE = "idle"

    # 8) Cleanup (but preserve embeddings until caller extracts them)
    del gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 9) Return decoded text **and** parsed JSON
    return decoded, data

def save_checkpoint(results_list, label_rows, selected_images):
    payload = {
        "version": "blank_v1",
        "selected_images": [
            {"filename": p.name, "img_path": str(p)}
            for p in selected_images
        ],
        "results": results_list,
    }
    np.save(CKPT_TEMP, payload)
    os.replace(CKPT_TEMP, CKPT_PATH)
    pd.DataFrame(label_rows).to_pickle(LABELS_DF_PATH)
    logger.info(f"Checkpoint saved → {CKPT_PATH} and {LABELS_DF_PATH}")

def get_selected_images(images_root: Path, max_images: int = 500):
    all_images = sorted(
        [
            p for p in images_root.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    if len(all_images) < max_images * 2:
        logger.warning(
            f"Expected about {max_images * 2} images for every-2nd selection, found only {len(all_images)}"
        )
    selected_images = all_images[::2][:max_images]
    logger.info(f"Found {len(all_images)} total images")
    logger.info(f"Selected {len(selected_images)} images using every 2nd image")
    for p in selected_images[:10]:
        logger.info(f"Selected preview: {p.name}")
    return selected_images

def _load_existing_results(path: str):
    if not os.path.exists(path):
        return []
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj.get("results", []) if isinstance(obj, dict) else []
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results from {path}: {e}")
        return []

# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def get_response_with_embeddings(image_path):
    prompt = f"""
        You see this image; rate its interestingness from:
        "Not Interesting","Slightly Interesting","Moderately Interesting",
        "Very Interesting","Extremely Interesting"

        Respond with JSON. Keys:
        - "interestingness": one of the five strings above
        - "explanation": 1–2 sentences
        """.strip()

    # Clear embeddings before processing
    layer_embeddings.clear()
    logger.info(f"🎯 Processing image: {image_path}")

    raw, data = model_response(prompt, image_path)

    # Log embedding capture results
    logger.info(f"📊 Captured embeddings: {len(layer_embeddings)} tensors")
    for name, tensor in layer_embeddings.items():
        logger.info(f"  {name}: {tuple(tensor.shape)}")

    # Convert to numpy for saving
    embeds = {k: remove_batch_dimension(v) for k,v in layer_embeddings.items()} if layer_embeddings else {}
    layer_embeddings.clear()  # Clear only after we've extracted the embeddings

    fname = os.path.basename(image_path)

    return {
        "filename": fname,
        "img_path": str(image_path),
        "interestingness_label": data["interestingness"],
        "explanation": data["explanation"],
        "embeddings": embeds,
    }


# ─── Cleanup function for hooks ──────────────────────────────────────────────────
def cleanup_hooks():
    global hook_handles
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()
    logger.info("Removed all hooks")


try:
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Missing images folder: {IMAGES_ROOT}")
    
    selected_images = get_selected_images(IMAGES_ROOT, MAX_IMAGES)
    existing = _load_existing_results(CKPT_PATH)
    results_list = list(existing)
    done_paths = {
            r.get("img_path") for r in results_list
        if isinstance(r, dict) and "img_path" in r
    }
    label_rows = [
            {
                "filename": r["filename"],
            "interestingness_label": r["interestingness_label"],
            "explanation": r["explanation"],
        }
        for r in results_list
        if isinstance(r, dict)
        and all(k in r for k in ["filename", "interestingness_label", "explanation"])
    ]
    logger.info(f"Resuming at {len(done_paths)}/{len(selected_images)} images")
    processed_count = 0
    for idx, img_path in enumerate(selected_images, start=1):
        img_path_str = str(img_path)
        if img_path_str in done_paths:
            continue
        try:
            logger.info(f"Processing image {idx}/{len(selected_images)}: {img_path_str}")
            out = get_response_with_embeddings(img_path_str)
            rec = dict(out) if isinstance(out, dict) else {"raw": out}
            results_list.append(rec)
            done_paths.add(img_path_str)
            processed_count += 1
            label_rows.append({
                    "filename": rec["filename"],
                "interestingness_label": rec["interestingness_label"],
                "explanation": rec["explanation"],
            })
            if len(done_paths) % CHECKPOINT_EVERY == 0 or len(done_paths) == len(selected_images):
                save_checkpoint(results_list, label_rows, selected_images)
        except Exception as e:
            logger.warning(f"❌ Error processing {img_path_str}: {e}; skipping")
            emergency_cleanup(preserve_embeddings=False)
            continue
    save_checkpoint(results_list, label_rows, selected_images)
    logger.info(f"Processed {processed_count} new images → ✅ {CKPT_PATH}")
except Exception as e:
    logger.error(f"Fatal error: {e}")
finally:
    cleanup_hooks()