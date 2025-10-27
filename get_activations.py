#!/usr/bin/env python
import os, sys, logging, argparse, json, re, gc, time

# ─── CUDA Memory Debugging ──────────────────────────────────────────────────────
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoImageProcessor,
    AutoModelForImageTextToText,
)

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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

# ─── HF local path hack ──────────────────────────────────────────────────────────
DEVROOT = "/anvme/workspace/iwi5268h-bigdata/dev"
sys.path.insert(0, f"{DEVROOT}/hf_deps")
sys.path.insert(0, f"{DEVROOT}/transformers/src")
import huggingface_hub.utils._validators as _hf_val
_hf_val.validate_repo_id = lambda *a, **k: None
import transformers.utils.hub as _tfh
_tfh.validate_repo_id = lambda *a, **k: None

# ─── Argument Parser ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

# ─── Paths & Checkpoint ──────────────────────────────────────────────────────────
CKPT_PATH = "data/results-FAU_25.npy"
CKPT_TEMP = "data/results-temp.npy"
OFFLOAD_DIR = "./offload_dir"
os.makedirs("data", exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

N_MAX_PERSONAS = 10 
SELECTED_25_PATH = "data/selected_uniform_25.pkl"
IMAGES_ROOT = "data/imagesDemographics"


results_list = []
try:
    if os.path.isfile(CKPT_PATH):
        ckpt = np.load(CKPT_PATH, allow_pickle=True).item()
        results_list = ckpt.get("results", [])
        logger.info(f"Resuming with {len(results_list)} entries from checkpoint")
    else:
        logger.info("Starting fresh (no checkpoint found)")
except Exception as e:
    logger.warning(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")

seen = set()
deduped = []
for r in results_list:
    key = (r["user_id"], r["img_id"])
    if key not in seen:
        deduped.append(r)
        seen.add(key)
results_list = deduped
done_pairs = set(seen)
counter = len(results_list)

# ─── Load model & processors ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
)

# Memory optimizations
model.eval()
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

logger.info("▶️ device map:")
for k,v in model.hf_device_map.items():
    logger.info(f"  {k} → {v}")
print_cuda_memory("▶️ initial")

processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)

# ─── Enhanced Model Inspection ─────────────────────────────────────────────────────
def inspect_model_structure():
    """Inspect the actual model structure to find hookable layers"""
    logger.info("🔍 Inspecting model structure...")
    
    # Check language model layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'layers'):
            logger.info(f"Found {len(model.language_model.model.layers)} language layers")
            for i, layer in enumerate(model.language_model.model.layers[:3]):  # Just first 3
                logger.info(f"  Language layer {i}: {type(layer)} on device {next(layer.parameters()).device if list(layer.parameters()) else 'no params'}")
    
    # Check vision model - try different access patterns
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
                for i, layer in enumerate(layers[:3]):  # Just first 3
                    logger.info(f"  Vision layer {i}: {type(layer)} on device {next(layer.parameters()).device if list(layer.parameters()) else 'no params'}")
                break
        except Exception as e:
            logger.debug(f"Path {path_name} failed: {e}")
    
    if vision_layers is None:
        logger.warning("⚠️ Could not find vision layers - checking vision_model structure")
        if hasattr(model, 'vision_model'):
            for attr in dir(model.vision_model):
                if not attr.startswith('_'):
                    logger.info(f"  vision_model.{attr}: {type(getattr(model.vision_model, attr))}")
    
    return vision_layers, vision_path_used

# ─── Hooks with Enhanced Debugging ────────────────────────────────────────────────
layer_embeddings = {}
hook_handles = []
hook_call_count = {}

def get_hook(name, store_dict):
    def hook(module, input, output):
        global hook_call_count
        hook_call_count[name] = hook_call_count.get(name, 0) + 1
        
        try:
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            
            # Only store if we have actual data
            if out is not None and hasattr(out, 'detach'):
                # Move to CPU immediately and store as float32
                store_dict[name] = out.detach().float().cpu()
                #logger.debug(f"🎣 Hook {name} captured: {out.shape} (call #{hook_call_count[name]})")
                del out
            else:
                logger.debug(f"🎣 Hook {name} - no valid output (call #{hook_call_count[name]})")
            
        except Exception as e:
            logger.warning(f"⚠️ Hook failed on {name}: {e}")
    return hook

# Inspect and register hooks
vision_layers, vision_path = inspect_model_structure()

# Register language model hooks
if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
    for i, layer in enumerate(model.language_model.model.layers):
        handle = layer.register_forward_hook(get_hook(f"language_layer_{i}", layer_embeddings))
        hook_handles.append(handle)
    logger.info(f"Registered {len(model.language_model.model.layers)} language hooks")

# Register vision model hooks
if vision_layers is not None:
    for i, layer in enumerate(vision_layers):
        handle = layer.register_forward_hook(get_hook(f"vision_layer_{i}", layer_embeddings))
        hook_handles.append(handle)
    logger.info(f"Registered {len(vision_layers)} vision hooks using path: {vision_path}")
else:
    logger.warning("⚠️ No vision layers found for hooking")

logger.info(f"Total hooks registered: {len(hook_handles)}")

def remove_batch_dimension(t):
    """Remove batch dimension from tensor and convert to numpy"""
    try:
        # Convert to CPU and float32 first
        cpu_tensor = t.to(torch.float32).cpu()
        
        # Remove the first dimension (batch dimension) if it exists and has size > 1
        if cpu_tensor.dim() > 0:
            if cpu_tensor.shape[0] == 1:
                # If batch size is 1, we can squeeze the first dimension
                unbatched = cpu_tensor.squeeze(0)
            else:
                # If batch size > 1, just take the first item
                unbatched = cpu_tensor[0]
        else:
            unbatched = cpu_tensor
        
        # Convert to numpy
        return unbatched.numpy()
        
    except Exception as e:
        logger.warning(f"Error in remove_batch_dimension: {e}, tensor shape: {t.shape}")
        # Fallback: just convert to numpy without manipulation
        return t.to(torch.float32).cpu().numpy()

def move_to_device(inputs, target_device):
    """Move all tensors in inputs to target_device"""
    moved_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved_inputs[key] = value.to(target_device)
            logger.debug(f"Moved {key} {value.shape} to {target_device}")
        else:
            moved_inputs[key] = value
    return moved_inputs

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

# ─── Core Response Function with Hook Debugging ──────────────────────────────────
def model_response(prompt, image_path, max_length=1000):
    global hook_call_count
    
    # 0) clear any leftover GPU scratch and reset hook counters (but preserve any existing embeddings from retries)
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

    # 2) Move inputs to correct device BEFORE filtering
    target_device = next(model.parameters()).device  # Get model's actual device
    inputs = move_to_device(inputs, target_device)
    
    # 3) Log what we're working with
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            logger.debug(f"Input: {k} {tuple(v.shape)} {v.dtype} @ {v.device}")

    # 4) Filter to only what the model expects
    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    extras = set(inputs) - whitelist
    if extras:
        logger.debug(f"Dropping unused inputs: {extras}")
    model_inputs = {k: inputs[k] for k in whitelist if k in inputs}

    # 5) Generate with progressive fallback
    generation_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.9,
        "use_cache": False,
        "do_sample": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    
    logger.info("🚀 Starting generation...")
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
            
            try:
                with torch.no_grad():
                    gen_ids = model.generate(**model_inputs, **generation_kwargs)
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    logger.warning("⚠️ Still OOM – retrying with 64 tokens")
                    emergency_cleanup()
                    hook_call_count.clear()
                    generation_kwargs["max_new_tokens"] = 64
                    with torch.no_grad():
                        gen_ids = model.generate(**model_inputs, **generation_kwargs)
                else:
                    raise e2
        else:
            raise

    # 6) Report hook activity
    logger.info("🎣 Hook call summary:")
    for name, count in hook_call_count.items():
        logger.info(f"  {name}: {count} calls")
    
    if not hook_call_count:
        logger.warning("⚠️ NO HOOKS WERE CALLED during generation!")
    
    # 7) Decode response
    out = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    if not out.strip() or "NaN" in out:
        logger.warning(f"⚠️ Suspect output: {out!r}")

    # 8) Cleanup (but preserve embeddings!)
    del inputs, model_inputs, gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 9) return assistant slice
    return out[out.find("assistant"):] if "assistant" in out else out

def _persona_ckpt_paths(persona_index: int):
    """
    Build per-persona checkpoint and temp paths.
    """
    ckpt = f"data/results-FAU_25_{persona_index}.npy"   # or results-FAU_100_... if you prefer
    tmp  = f"data/results-temp_{persona_index}.npy"
    return ckpt, tmp

def _load_existing_results(path: str):
    """
    Load an existing checkpoint's 'results' list, or [] if none/corrupt.
    """
    if not os.path.exists(path):
        return []
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj.get("results", []) if isinstance(obj, dict) else []
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results from {path}: {e}")
        return []


# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def get_response_with_embeddings(user_id, df_temp, image_path):
    cols = ['age','gender','country','continent','job_branch','mental_workload','emotion']
    persona = ", ".join(str(df_temp[c].iloc[0]) for c in cols)
    prompt = f"""
        Imagine you are a person with the following details:
        {persona}

        You see this image; rate its interestingness from:
        "Not Interesting","Slightly Interesting","Moderately Interesting",
        "Very Interesting","Extremely Interesting"
        Respond with JSON:
        {{"interestingness":"...","explanation":"..."}}
        """
    
    # Clear embeddings before processing
    layer_embeddings.clear()
    logger.info(f"🎯 Processing image: {image_path}")
    
    raw = model_response(prompt, image_path)
    
    # Log embedding capture results
    logger.info(f"📊 Captured embeddings: {len(layer_embeddings)} layers")
    for name in layer_embeddings.keys():
        shape = layer_embeddings[name].shape
        logger.info(f"  {name}: {shape}")
    
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        logger.warning(f"No JSON found in response: {raw[:200]}...")
        raise ValueError("No JSON in response")
    
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}, raw: {m.group()}")
        raise ValueError("Invalid JSON in response")
    
    if not layer_embeddings:
        logger.warning("⚠️ No embeddings recorded - check hook registration and model execution path")
        # Still return the result but with empty embeddings
        embeds = {}
    else:
        embeds = {k: remove_batch_dimension(v) for k,v in layer_embeddings.items()}
    
    layer_embeddings.clear()  # Clear only after we've extracted the embeddings

    fname = os.path.basename(image_path)
    img_id = os.path.splitext(fname)[0]

    return {"user_id":user_id, "img_id":img_id, **data, "embeddings":embeds}

# ─── Cleanup function for hooks ──────────────────────────────────────────────────
def cleanup_hooks():
    """Remove all registered hooks"""
    global hook_handles
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()
    logger.info("Removed all hooks")

try:
    # Personas
    df_personas = pd.read_pickle("data/df_generated-personas-sample.pkl")
    persona_ids = list(df_personas.index)[:N_MAX_PERSONAS]
    logger.info(f"Running {len(persona_ids)} personas")

    # Load fixed 25 filenames and build img_path
    if not os.path.exists(SELECTED_25_PATH):
        raise FileNotFoundError(f"Missing {SELECTED_25_PATH}. Run selection_utils.py first.")
    df_selected = pd.read_pickle(SELECTED_25_PATH).copy()
    if "filename" not in df_selected.columns:
        raise ValueError("selected_uniform_25.pkl must contain a 'filename' column.")

    df_selected["img_path"] = df_selected["filename"].apply(lambda f: os.path.join(IMAGES_ROOT, f))
    df_selected = df_selected.drop_duplicates(subset=["img_path"]).head(25).reset_index(drop=True)

    for persona_index, user in enumerate(persona_ids):
        persona_row = df_personas.loc[[user]]
        CKPT_PATH, CKPT_TEMP = _persona_ckpt_paths(persona_index)

        # Resume support per persona
        existing = _load_existing_results(CKPT_PATH)
        done_paths = {r.get("img_path") for r in existing if isinstance(r, dict) and "img_path" in r}
        results_list = list(existing)
        counter = len(results_list)
        processed_count = 0

        logger.info(f"[persona {persona_index}] {user} → resuming at {counter}/25")

        for _, srow in df_selected.iterrows():
            img_path = srow["img_path"]
            if img_path in done_paths:
                continue
            try:
                logger.info(f"Processing image: {img_path}")
                out = get_response_with_embeddings(user, persona_row, img_path)

                rec = dict(out) if isinstance(out, dict) else {"raw": out}
                rec.setdefault("img_path", img_path)
                rec.setdefault("filename", srow["filename"])
                rec.setdefault("persona_id", user)
                rec.setdefault("persona_index", persona_index)
                if "interestingness_label" in srow and "interestingness_label" not in rec:
                    rec["interestingness_label"] = srow["interestingness_label"]

                results_list.append(rec)
                done_paths.add(img_path)
                counter += 1
                processed_count += 1

                # Checkpoint frequently
                if counter % 2 == 0 or counter == 25:
                    payload = {
                        "version": "v1",
                        "persona_id": user,
                        "persona_index": persona_index,
                        "selected_images": df_selected.to_dict(orient="records"),
                        "results": results_list,
                    }
                    np.save(CKPT_TEMP, payload)
                    os.replace(CKPT_TEMP, CKPT_PATH)
                    logger.info(f"[persona {persona_index}] checkpoint → {CKPT_PATH}")

            except Exception as e:
                logger.warning(f"❌ Error processing {img_path}: {e}; skipping")
                emergency_cleanup(preserve_embeddings=False)
                continue

        # Final save for this persona
        payload = {
            "version": "v1",
            "persona_id": user,
            "persona_index": persona_index,
            "selected_images": df_selected.to_dict(orient="records"),
            "results": results_list,
        }
        np.save(CKPT_TEMP, payload)
        os.replace(CKPT_TEMP, CKPT_PATH)
        logger.info(f"[persona {persona_index}] Processed {processed_count} new images → ✅ {CKPT_PATH}")

except Exception as e:
    logger.error(f"Fatal error: {e}")

finally:
    cleanup_hooks()

"""# ─── Run single user for debug ───────────────────────────────────────────────────
try:
    df_personas = pd.read_pickle("data/df_generated-personas-sample.pkl")
    df_images   = pd.read_pickle("data/df_common_machine_int.pkl")[['img_path','interestingness_value']]
    logger.info(f"{len(df_personas)} personas; {len(df_images)} images total")

    user = df_personas.index[2]
    processed_count = 0
    
    sampled_image_paths = df_images.iloc[::10].img_path

    for img_path in sampled_image_paths:
        key = (user, df_images[df_images.img_path==img_path].index[0])
        if key in done_pairs: 
            continue
            
        try:
            logger.info(f"Processing image: {img_path}")
            out = get_response_with_embeddings(user, df_personas.loc[[user]], img_path, df_images)
            results_list.append(out)
            done_pairs.add(key)
            counter += 1
            processed_count += 1
            
            logger.info(f"[{counter}] → {out['interestingness']}")
            
            # Checkpoint more frequently to avoid losing work
            if counter % 2 == 0:
                logger.info(f"Checkpointing at {counter}")
                np.save(CKPT_TEMP, {"results": results_list})
                os.replace(CKPT_TEMP, CKPT_PATH)
                
        except Exception as e:
            logger.warning(f"❌ Error processing {img_path}: {e}; skipping")
            emergency_cleanup(preserve_embeddings=False)  # Full cleanup on error
            continue

    # Final save
    logger.info(f"Processed {processed_count} new images")
    np.save(CKPT_TEMP, {'results': results_list})
    os.replace(CKPT_TEMP, CKPT_PATH)
    logger.info("✅ Done.")

except Exception as e:
    logger.error(f"Fatal error: {e}")
    # Save whatever we have
    if results_list:
        np.save(CKPT_TEMP, {'results': results_list})
        os.replace(CKPT_TEMP, CKPT_PATH)
        logger.info(f"Emergency save completed with {len(results_list)} results")

finally:
    cleanup_hooks()"""

"""
# Full processing loop (commented out for single-user debug)
for user in df_personas.index:
    for img_path in df_images.img_path:
        key = (user, df_images.index[df_images.img_path==img_path][0])
        if key in done_pairs:
            continue
        try:
            out = get_response_with_embeddings(user, df_personas.loc[[user]], img_path, df_images)
            results_list.append(out)
            done_pairs.add(key)
            counter += 1
            logger.info(f"[{counter}] user={user}, img={key[1]} → {out['interestingness']}")
            if counter % 5 == 0:
                logger.info(f"Checkpointing at {counter}")
                np.save(CKPT_TEMP, {'results': results_list})
                os.replace(CKPT_TEMP, CKPT_PATH)  # atomic
        except Exception as e:
            logger.warning(f"user={user}, img={key[1]} error: {e}; skipping")
            emergency_cleanup()
"""