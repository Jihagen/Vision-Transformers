
import os


# ─── CUDA Memory Debugging ──────────────────────────────────────────────────────

os.environ["HF_HOME"] = r"D:\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"
os.environ["HF_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face API token.")


repo_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

path = snapshot_download(
    repo_id=repo_id,
    resume_download=True,
    token=hf_token,
)
print("Downloaded to:", path)
