#!/usr/bin/env python
"""
Rate interestingness for ALL images in a folder.
Saves a Pandas DataFrame with columns: ["filename", "interestingness"].

Example:
python rate.py \
  --model_path /path/to/local/model \
  --images_dir /path/to/images \
  --local_only
"""
import os, sys, logging, argparse, json, re, gc, time, glob

# ─── CUDA Memory ────────────────────────────────────────────────────────────────
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
IMAGES_DIR = "data"  # hardcoded image folder

# ─── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Rate image interestingness (JSON only).")
parser.add_argument("--model_path", required=True, help="Local path to HF model")
parser.add_argument("--output", default="data/interestingness_results.pkl",
                    help="Output DataFrame pickle (.pkl). CSV is written alongside automatically.")
parser.add_argument("--pattern", default="**/*", help="Glob pattern under images_dir (recursive)")
parser.add_argument("--max_side", type=int, default=800, help="Max image side for resizing")
parser.add_argument("--local_only", action="store_true", help="Use local files only for model/proc")
parser.add_argument("--ckpt_every", type=int, default=10, help="Checkpoint every N processed images")
args = parser.parse_args()

# ─── Paths & Setup ──────────────────────────────────────────────────────────────
out_dir = os.path.dirname(args.output) or "."
os.makedirs(out_dir, exist_ok=True)
offload_dir = "./offload_dir"
os.makedirs(offload_dir, exist_ok=True)

csv_path = args.output[:-4] + ".csv" if args.output.endswith(".pkl") else args.output + ".csv"
tmp_path = args.output + ".tmp"

# ─── Helpers ────────────────────────────────────────────────────────────────────
def emergency_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    time.sleep(0.1)

def preprocess_image(image_path, max_size=800):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        image = image.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)
    return image

def decode_interestingness(text):
    allowed = {
        "Not Interesting",
        "Slightly Interesting",
        "Moderately Interesting",
        "Very Interesting",
        "Extremely Interesting",
    }

    # Try all JSON-like snippets, pick the first valid one
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            data = json.loads(m.group())
        except Exception:
            continue
        val = data.get("interestingness")
        if isinstance(val, str) and val.strip() in allowed:
            return val.strip()

    # Fallback: try to find a bare label in the text
    for lab in allowed:
        if lab in text:
            return lab

    raise ValueError(f"No valid interestingness label found in: {text!r}")

def build_prompt():
    return (
        "You are a strict JSON generator.\n"
        "Task: Rate the interestingness of THIS IMAGE using exactly ONE of:\n"
        "\"Not Interesting\", \"Slightly Interesting\", \"Moderately Interesting\", "
        "\"Very Interesting\", \"Extremely Interesting\".\n\n"
        "Rubric:\n"
        "- Not Interesting: plain/typical, no notable subject/composition.\n"
        "- Slightly Interesting: one minor point of interest.\n"
        "- Moderately Interesting: clear subject or composition; a few notable elements.\n"
        "- Very Interesting: strong subject, composition, or moment; multiple notable elements.\n"
        "- Extremely Interesting: striking/rare/exceptional; highly memorable.\n\n"
        "Important:\n"
        "Output ONLY a single JSON object like:\n"
        "{ \"interestingness\": \"Moderately Interesting\" }"
    )


def pick_pad_token_id(processor, model):
    # Be robust if eos_token_id is missing
    tok = getattr(processor, "tokenizer", None)
    for attr in ["eos_token_id", "pad_token_id"]:
        if tok is not None and getattr(tok, attr, None) is not None:
            return getattr(tok, attr)
    for attr in ["eos_token_id", "pad_token_id"]:
        if getattr(model.config, attr, None) is not None:
            return getattr(model.config, attr)
    return None

def generate_for_image(model, processor, image_path, max_new_tokens=256):
    prompt = build_prompt()
    image = preprocess_image(image_path, max_size=args.max_side)

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

    # Move tensors to the model's first parameter device
    target_device = next(model.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(target_device, non_blocking=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        do_sample=False,
        use_cache=False,
    )
    pad_id = pick_pad_token_id(processor, model)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    # ⬇️ Keep only the new tokens after the input length
    gen_only = out_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
    return decode_interestingness(text)


# ─── Load model & processor ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    local_files_only=args.local_only,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    offload_folder=offload_dir,
    offload_state_dict=True,
)
model.eval()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_only)

# ─── Discover images ────────────────────────────────────────────────────────────
exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
all_files = [
    p for p in glob.glob(os.path.join(IMAGES_DIR, args.pattern), recursive=True)
    if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
]
all_files.sort()
logger.info(f"Found {len(all_files)} images under {IMAGES_DIR}")

# ─── Resume DataFrame (if present) ──────────────────────────────────────────────
if os.path.isfile(args.output):
    try:
        df_results = pd.read_pickle(args.output)
        # Ensure expected columns
        if not {"filename","interestingness"}.issubset(df_results.columns):
            logger.warning("Existing pickle lacks required columns; starting fresh.")
            df_results = pd.DataFrame(columns=["filename", "interestingness"])
        logger.info(f"Resuming with {len(df_results)} prior rows from {args.output}")
    except Exception as e:
        logger.warning(f"Could not read existing DataFrame ({e}); starting fresh.")
        df_results = pd.DataFrame(columns=["filename", "interestingness"])
else:
    df_results = pd.DataFrame(columns=["filename", "interestingness"])

done_names = set(df_results["filename"].astype(str))

# ─── Main loop ──────────────────────────────────────────────────────────────────
processed = 0
total = len(all_files)
for i, img_path in enumerate(all_files, 1):
    fname = os.path.basename(img_path)
    if fname in done_names:
        continue

    try:
        logger.info(f"[{i}/{total}] {img_path}")
        try_tokens = [256, 128, 64]  # graceful OOM fallback
        last_err = None
        for tks in try_tokens:
            try:
                rating = generate_for_image(model, processor, img_path, max_new_tokens=tks)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at {tks} tokens, retrying with fewer…")
                    emergency_cleanup()
                    last_err = e
                    continue
                else:
                    last_err = e
                    raise
        else:
            raise last_err or RuntimeError("Unknown generation failure")

        # Append to DataFrame
        df_results = pd.concat(
            [df_results, pd.DataFrame([{"filename": fname, "interestingness": rating}])],
            ignore_index=True
        )
        done_names.add(fname)
        processed += 1

        # Checkpoint every N new rows
        if processed % args.ckpt_every == 0:
            df_results.to_pickle(tmp_path)
            os.replace(tmp_path, args.output)
            df_results.to_csv(csv_path, index=False)
            logger.info(f"Checkpointed {processed} new rows → {args.output} (+ CSV)")

    except Exception as e:
        logger.warning(f"Skipping {img_path} due to error: {e}")
        emergency_cleanup()
        continue

# ─── Final save ─────────────────────────────────────────────────────────────────
df_results.to_pickle(tmp_path)
os.replace(tmp_path, args.output)
df_results.to_csv(csv_path, index=False)
logger.info(f"✅ Done. Saved {len(df_results)} rows to {args.output} and {csv_path}")
