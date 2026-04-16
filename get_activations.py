import os, sys, logging, argparse, json, re, gc 
from typing import Optional


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── local HF/dev path (fallback) ────────────────────────────────────────────
DEVROOT = os.environ.get("DEVROOT", "/anvme/workspace/iwi5268h-vision-transformers/dev")
HF_DEPS = os.path.join(DEVROOT, "hf_deps")
TF_SRC = os.path.join(DEVROOT, "transformers", "src")

if os.path.isdir(HF_DEPS):
    sys.path.insert(0, HF_DEPS)
if os.path.isdir(TF_SRC):
    sys.path.insert(0, TF_SRC)

try:
    import huggingface_hub.utils._validators as _hf_val
    _hf_val.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    import transformers.utils.hub as _tfh
    _tfh.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

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
    """Best-effort memory cleanup; safe to call even when CUDA context is broken."""
    global layer_embeddings, hook_handles
    if not preserve_embeddings:
        layer_embeddings.clear()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as _ce:
            logger.warning(f"⚠️ emergency_cleanup: torch.cuda.empty_cache() failed: {_ce}")
        try:
            torch.cuda.ipc_collect()
        except Exception as _ci:
            logger.warning(f"⚠️ emergency_cleanup: torch.cuda.ipc_collect() failed: {_ci}")
    gc.collect()



# ─── Argument Parser ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

# ─── Paths & Checkpoint ──────────────────────────────────────────────────────────
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "./offload_dir")
os.makedirs("data", exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

CHECKPOINT_EVERY = 2
MAX_RETRY_ROUNDS = 5
MANIFEST_PATH = "data/selected_uniform_500_manifest.pkl"
PERSONA_PATH = "data/df_generated-personas-sample.pkl"

# Final chosen personas
SELECTED_PERSONA_IDS = [8447, 3770, 7319, 5504]
N_MAX_PERSONAS = len(SELECTED_PERSONA_IDS)


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

model.eval()

logger.info("▶️ device map:")
if hasattr(model, "hf_device_map"):
    for k, v in model.hf_device_map.items():
        logger.info(f"  {k} → {v}")

print_cuda_memory("▶️ initial")

processor = AutoProcessor.from_pretrained(
    args.model_path,
    local_files_only=True,
)

# ─── Helpers for Data & Redo-logic ──────────────────────────────────────────────
def load_manifest(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_pickle(path)
    required = {"filename", "img_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns {required}, got {set(df.columns)}")
    return df.copy()

def get_missing_rows(manifest_df: pd.DataFrame, results_list: list[dict]) -> pd.DataFrame:
    done_files = {
        r.get("filename")
        for r in results_list
        if isinstance(r, dict) and "filename" in r
    }
    missing_df = manifest_df[~manifest_df["filename"].isin(done_files)].copy()
    return missing_df.reset_index(drop=True)


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


def model_response(prompt, image_path, image_max_size=800, max_new_tokens=64):
    global hook_call_count, CAPTURE_PHASE

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hook_call_count.clear()

    image = preprocess_image(image_path, max_size=image_max_size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
    text=chat_input,
    images=image,
    return_tensors="pt",
    add_special_tokens=True,
    truncation=False,
    max_length=512,
    )

    target_device = next(model.parameters()).device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(target_device)

    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    extras = set(inputs) - whitelist
    if extras:
        logger.debug(f"Dropping unused inputs: {extras}")
    model_inputs = {k: inputs[k] for k in whitelist if k in inputs}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    logger.info(
        f"🚀 Starting generation (vision capture on, image_max_size={image_max_size}, max_new_tokens={max_new_tokens})..."
    )
    CAPTURE_PHASE = "vision_once"
    try:
        print_cuda_memory("Before generate")
        with torch.no_grad():
            gen_ids = model.generate(**model_inputs, **generation_kwargs)
        print_cuda_memory("After generate")
    finally:
        CAPTURE_PHASE = "idle"

    decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    try:
        data = extract_last_json_block(decoded)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}; raw tail: {decoded[-400:]}")
        raise ValueError("Invalid JSON in response")

    full_seq_ids = gen_ids[0].tolist()
    input_len = model_inputs["input_ids"].shape[1]

    forced_anchor = '{"interestingness":"'
    forced_ids = processor.tokenizer.encode(forced_anchor, add_special_tokens=False)
    prefix_ids = full_seq_ids[:input_len] + forced_ids

    local = {
        "input_ids": torch.tensor([prefix_ids], device=model_inputs["input_ids"].device),
        "attention_mask": torch.ones(
            1,
            len(prefix_ids),
            device=model_inputs["attention_mask"].device,
            dtype=model_inputs["attention_mask"].dtype,
        ),
        "pixel_values": model_inputs["pixel_values"],
    }

    logger.info("🎯 Teacher-forced one-step to capture first label token (LLM layers)…")
    CAPTURE_PHASE = "rating_step"
    with torch.no_grad():
        _ = model.generate(
            **local,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    CAPTURE_PHASE = "idle"

    del gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return decoded, data

def _persona_ckpt_paths(persona_id):
    ckpt = f"data/results_persona_{persona_id}.npy"
    tmp  = f"data/results_persona_{persona_id}_temp.npy"
    return ckpt, tmp

def _load_existing_results(path: str):
    if not os.path.exists(path):
        return []
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj.get("results", []) if isinstance(obj, dict) else []
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results from {path}: {e}")
        return []
    
def save_persona_checkpoint(results_list, manifest_df: pd.DataFrame, persona_id, persona_index, ckpt_temp, ckpt_path):
    payload = {
        "version": "persona_v1",
        "persona_id": persona_id,
        "persona_index": persona_index,
        "selected_images": manifest_df[["filename", "img_path"]].to_dict(orient="records"),
        "results": results_list,
    }
    np.save(ckpt_temp, payload)
    os.replace(ckpt_temp, ckpt_path)
    logger.info(f"[persona {persona_index}] checkpoint → {ckpt_path}")

# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def get_response_with_embeddings(user_id, df_temp, image_path):
    cols = ['age', 'gender', 'country', 'continent', 'job_branch', 'mental_workload', 'emotion']
    persona = ", ".join(str(df_temp[c].iloc[0]) for c in cols)

    prompt = f"""
        Imagine you are a person with the following details:
        {persona}

        You see this image; rate its interestingness from:
        "Not Interesting","Slightly Interesting","Moderately Interesting",
        "Very Interesting","Extremely Interesting"

        Return ONLY valid JSON.
        Do not use markdown.
        Do not use code fences.
        Do not add any text before or after the JSON.

        Keys:
        - "interestingness": one of the five strings above
        - "explanation": exactly 1 short sentence
        """.strip()

    layer_embeddings.clear()
    logger.info(f"🎯 Processing image: {image_path}")

    try:
        raw, data = model_response(
            prompt,
            image_path,
            image_max_size=800,
            max_new_tokens=64,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom_sizes = [640, 512, 384, 256, 128]
            last_exc = e
            raw = data = None
            for oom_size in oom_sizes:
                logger.warning(f"⚠️ OOM for {image_path}; retrying with image_max_size={oom_size}.")
                emergency_cleanup(preserve_embeddings=False)
                try:
                    raw, data = model_response(
                        prompt,
                        image_path,
                        image_max_size=oom_size,
                        max_new_tokens=64,
                    )
                    break
                except RuntimeError as e_retry:
                    if "illegal memory access" in str(e_retry).lower():
                        logger.error(
                            f"❌ CUDA illegal memory access during OOM retry for {image_path}; "
                            "CUDA context is corrupted — aborting."
                        )
                        raise
                    if "out of memory" in str(e_retry).lower():
                        last_exc = e_retry
                        continue
                    raise
            else:
                logger.error(f"❌ OOM persists for {image_path} even at image_max_size={oom_sizes[-1]}; skipping.")
                raise last_exc
        else:
            raise

    except ValueError as e:
        if "Invalid JSON" in str(e):
            logger.warning(
                f"⚠️ Invalid JSON for {image_path} at max_new_tokens=64; retrying with more output tokens."
            )
            emergency_cleanup(preserve_embeddings=False)

            try:
                raw, data = model_response(
                    prompt,
                    image_path,
                    image_max_size=800,
                    max_new_tokens=96,
                )
            except ValueError as e2:
                if "Invalid JSON" in str(e2):
                    logger.warning(
                        f"⚠️ Invalid JSON for {image_path} at max_new_tokens=96; retrying with more output tokens."
                    )
                    emergency_cleanup(preserve_embeddings=False)

                    raw, data = model_response(
                        prompt,
                        image_path,
                        image_max_size=800,
                        max_new_tokens=128,
                    )
                else:
                    raise
        else:
            raise

    logger.info(f"📊 Captured embeddings: {len(layer_embeddings)} tensors")
    for name, tensor in layer_embeddings.items():
        logger.info(f"  {name}: {tuple(tensor.shape)}")

    embeds = {k: remove_batch_dimension(v) for k, v in layer_embeddings.items()} if layer_embeddings else {}
    layer_embeddings.clear()

    fname = os.path.basename(image_path)
    img_id = os.path.splitext(fname)[0]

    return {
        "user_id": user_id,
        "img_id": img_id,
        "filename": fname,
        "img_path": str(image_path),
        "interestingness": data["interestingness"],
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
    manifest_df = load_manifest(MANIFEST_PATH)
    logger.info(f"Loaded manifest with {len(manifest_df)} target images")

    df_personas = pd.read_pickle(PERSONA_PATH)

    persona_ids = SELECTED_PERSONA_IDS[:N_MAX_PERSONAS]
    logger.info(f"Running {len(persona_ids)} personas: {persona_ids}")

    for persona_index, user in enumerate(persona_ids):
        if user not in df_personas.index:
            raise ValueError(f"Persona id {user} not found in persona dataframe")

        persona_row = df_personas.loc[[user]]
        CKPT_PATH, CKPT_TEMP = _persona_ckpt_paths(user)

        existing = _load_existing_results(CKPT_PATH)
        results_list = list(existing)

        logger.info(f"[persona {persona_index}] persona_id={user}")
        logger.info(f"[persona {persona_index}] existing successful results: {len(results_list)}")

        total_processed_new = 0

        for retry_round in range(MAX_RETRY_ROUNDS + 1):
            missing_df = get_missing_rows(manifest_df, results_list)
            n_missing = len(missing_df)

            if n_missing == 0:
                logger.info(f"[persona {persona_index}] ✅ All target images processed.")
                break

            logger.info(
                f"[persona {persona_index}] === Retry round {retry_round}/{MAX_RETRY_ROUNDS} | missing: {n_missing} ==="
            )
            processed_this_round = 0

            for idx, row in enumerate(missing_df.itertuples(index=False), start=1):
                img_path_str = str(row.img_path)

                try:
                    logger.info(f"[persona {persona_index}] [round {retry_round}] Processing {idx}/{n_missing}: {img_path_str}")
                    out = get_response_with_embeddings(user, persona_row, img_path_str)
                    rec = dict(out) if isinstance(out, dict) else {"raw": out}

                    rec.setdefault("persona_id", user)
                    rec.setdefault("persona_index", persona_index)

                    results_list.append(rec)
                    total_processed_new += 1
                    processed_this_round += 1

                    if len(results_list) % CHECKPOINT_EVERY == 0:
                        save_persona_checkpoint(
                            results_list,
                            manifest_df,
                            user,
                            persona_index,
                            CKPT_TEMP,
                            CKPT_PATH,
                        )

                except Exception as e:
                    if "illegal memory access" in str(e).lower():
                        logger.error(
                            f"[persona {persona_index}] ❌ CUDA illegal memory access on {img_path_str}; "
                            "CUDA context is corrupted — aborting run."
                        )
                        raise
                    logger.warning(f"[persona {persona_index}] ❌ Error processing {img_path_str}: {e}; skipping for now")
                    emergency_cleanup(preserve_embeddings=False)
                    continue

            save_persona_checkpoint(
                results_list,
                manifest_df,
                user,
                persona_index,
                CKPT_TEMP,
                CKPT_PATH,
            )
            logger.info(
                f"[persona {persona_index}] Round {retry_round} finished; newly processed this round: {processed_this_round}"
            )

            if processed_this_round == 0:
                logger.warning(f"[persona {persona_index}] No progress in this retry round; stopping retries early.")
                break

        final_missing_df = get_missing_rows(manifest_df, results_list)
        logger.info(f"[persona {persona_index}] Processed {total_processed_new} new images total")
        logger.info(f"[persona {persona_index}] Remaining missing after retries: {len(final_missing_df)}")

        if len(final_missing_df) > 0:
            missing_path = f"data/missing_persona_{user}.pkl"
            final_missing_df.to_pickle(missing_path)
            logger.warning(f"[persona {persona_index}] Saved remaining missing rows to {missing_path}")

except KeyboardInterrupt:
    logger.warning("Interrupted by user.")
    raise

except Exception as e:
    logger.error(f"Fatal error: {e}")

finally:
    cleanup_hooks()