import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict, Any
from einops import einsum, rearrange
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

def get_layer_cache_path(base_dir: str, dataset_name: str, layer_idx: int) -> str:
    """Standardizes the file path construction."""
    return os.path.join(base_dir, dataset_name, f"layer_{layer_idx}.pt")

def check_layer_cache(base_dir: str, dataset_name: str, layer_idx: int) -> bool:
    """Checks if a specific layer's cache exists."""
    path = get_layer_cache_path(base_dir, dataset_name, layer_idx)
    return os.path.exists(path)

def save_layer_cache(base_dir: str, dataset_name: str, layer_idx: int, data: Dict):
    """Saves a single layer's data to disk."""
    path = get_layer_cache_path(base_dir, dataset_name, layer_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    print(f"Saved cache: {path}")

def load_layer_cache(base_dir: str, dataset_name: str, layer_idx: int, device="cpu") -> Dict:
    """Loads a single layer's data from disk."""
    path = get_layer_cache_path(base_dir, dataset_name, layer_idx)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found at {path}")
    return torch.load(path, map_location=device, weights_only=False)

def get_tau(weight, A, G):
    """Calculates tau = G * W * A and normalizes it."""
    w = weight.float()
    # A and G might be on CPU, ensure they are on the same device as w for calculation
    A = A.to(w.device)
    G = G.to(w.device)

    temp = einsum(w, A, "d_out d, d din -> d_out din")
    tau = einsum(G, temp, "d_out d, d d_in -> d_out d_in").to(weight.dtype)
    tau_norm = (tau.norm() + 1e-8)
    return tau / tau_norm

def load_model(path:str, device:str = "cuda"):
    """Loads model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map=device,
        trust_remote_code=True,
        dtype=torch.float16
    )
    return model, tokenizer

def validate_batch_size(args):
    if args.batch_size != 1:
        print(f"WARNING: Batch size {args.batch_size} requested, but logic requires 1.")
        print("Forcing batch_size = 1.")
        args.batch_size = 1
    return args

def fisher_overlap(A1, G1, A2, G2):
    A1, G1 = A1.float(), G1.float()
    A2, G2 = A2.float(), G2.float()
    tr_g12 = (G1 * G2).sum()
    tr_a12 = (A1 * A2).sum()
    tr_g11 = (G1 * G1).sum()
    tr_a11 = (A1 * A1).sum()
    norm1 = torch.sqrt(tr_g11 * tr_a11)
    tr_g22 = (G2 * G2).sum()
    tr_a22 = (A2 * A2).sum()
    norm2 = torch.sqrt(tr_g22 * tr_a22)
    similarity = tr_g12 * tr_a12 / (norm1 * norm2)
    return similarity.item()