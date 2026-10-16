"""
Repository-relative default locations for the model checkpoint and dev tree.

Nothing here is machine-specific: defaults resolve against the repository
root and can be overridden through environment variables.

    VT_HF_CACHE    directory holding the local model cache
                   (default: <repo>/hpc_infrastructure/hf_cache)
    VT_MODEL_PATH  directory of the model checkpoint itself
                   (default: <VT_HF_CACHE>/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo)
    DEVROOT        optional custom transformers build
                   (default: <repo>/hpc_infrastructure/dev)
"""

from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

HF_CACHE_DIR = os.environ.get(
    "VT_HF_CACHE", str(REPO_ROOT / "hpc_infrastructure" / "hf_cache")
)

LOCAL_MODEL_REPO = os.environ.get(
    "VT_MODEL_PATH",
    str(Path(HF_CACHE_DIR) / "models--meta-llama--Llama-4-Scout-17B-16E-Instruct" / "local-repo"),
)

DEVROOT = os.environ.get("DEVROOT", str(REPO_ROOT / "hpc_infrastructure" / "dev"))


def resolve_data_path(path: str | os.PathLike) -> str:
    """
    Return path if it exists; otherwise re-anchor its 'data/...' tail at the
    repository root. Saved manifests hold absolute paths from the machine and
    directory layout they were written on.
    """
    p = Path(path)
    if p.exists() or "data" not in p.parts:
        return str(p)
    tail = p.parts[len(p.parts) - 1 - p.parts[::-1].index("data"):]
    candidate = REPO_ROOT.joinpath(*tail)
    return str(candidate) if candidate.exists() else str(p)
