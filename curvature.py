import torch
import torch.nn as nn
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict, Any
from einops import einsum, rearrange
from tqdm import tqdm
from utils2 import *


class KFAC:
    """
    Collects Input Covariance A = E[x^T x] and Gradient Covariance G = E[g^T g].
    """
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.device = layer.weight.device
        d_out, d_in = layer.weight.shape
        self.A = torch.zeros(d_in, d_in, dtype=torch.float32, device=self.device)
        self.G = torch.zeros(d_out, d_out, dtype=torch.float32, device=self.device)
        self.n_samples = 0
        self._buf = None
        self._mask = None
        self._batch_mask = None
        self._h_fwd = layer.register_forward_pre_hook(self._fwd, prepend=False)
        self._h_bwd = layer.register_full_backward_hook(self._bwd, prepend=False)

    def set_batch_mask(self, mask: torch.Tensor):
        self._batch_mask = mask

    def _fwd(self, module: nn.Module, inp: Tuple[torch.Tensor]) -> None:
        if torch.is_grad_enabled():
            x = inp[0].detach()
            x = x[:, :-1, :]

            if self._batch_mask is not None:
                mask_slice = self._batch_mask[:, :-1].to(x.device)
                bool_mask = mask_slice > 0.5
                self._buf = x[bool_mask].float()
                self._active_mask = bool_mask
            else:
                self._buf = rearrange(x, "batch seq d_in -> (batch seq) d_in").float()
                self._mask = None

    def _bwd(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]) -> None:
        if grad_output[0] is None or self._buf is None:
            return
        g = grad_output[0].detach()
        g = g[:, :-1, :]
        if self._active_mask is not None:
            g_flat = g[self._active_mask].float()
        else:
            g_flat = rearrange(g, "batch seq d_out -> (batch seq) d_out").float()

        self.A.add_(einsum(self._buf, self._buf, "d d_in, d din-> d_in din"))
        self.G.add_(einsum(g_flat, g_flat, "d d_out, d dout -> d_out dout"))
        self.n_samples += g_flat.size(0)
        self._buf = None
        self._active_mask = None

    def get_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_samples == 0:
            return self.A, self.G
        return self.A / self.n_samples, self.G / self.n_samples

    def close(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()
        self._buf = None
        self._mask = None
        self._batch_mask = None


class Steering:
    def __init__(self, model, tokenizer, layer_indices: list, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.edits = layer_indices
        self._cache = {}

    def collect_stats(self, dataloader, num_batches: int, dataset_name: str, cache_dir: str = "./stats_cache"):
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

        for i in tqdm(self.edits, desc="Collecting Factors"):
            if check_layer_cache(cache_dir, dataset_name, i):
                print(f"Layer {i} found in cache. Loading...")
                self._cache[i] = load_layer_cache(cache_dir, dataset_name, i)
                continue

            lin = self.model.model.layers[i].mlp
            mods = {
                "gate": lin.gate_proj,
                "up": lin.up_proj,
                "down": lin.down_proj
            }
            self._cache[i] = {}

            for name, mod in mods.items():
                mod.weight.requires_grad = True
                collector = KFAC(mod)
                self.model.zero_grad()
                batch_iter = iter(dataloader)
                for _ in range(num_batches):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(dataloader)
                        batch = next(batch_iter)
                    target_device = self.model.device
                    input_ids = batch['input_ids'].to(target_device)
                    attention_mask = batch['attention_mask'].to(target_device)
                    labels = batch.get('labels', None)
                    if labels is not None:
                        labels = labels.to(target_device)

                    assert input_ids.shape[0] == 1, f"Batch size must be 1, got {input_ids.shape[0]}"

                    collector.set_batch_mask(attention_mask)
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    self.model.zero_grad()

                A, G = collector.get_factors()
                self._cache[i][name] = (A.detach().cpu(), G.detach().cpu())
                collector.close()
                mod.weight.requires_grad = False
                del A, G, outputs, loss, input_ids, attention_mask
                torch.cuda.empty_cache()
                gc.collect()

            save_layer_cache(cache_dir, dataset_name, i, self._cache[i])

    def edit_layers(self, alpha=1.0):
        assert len(self._cache) > 0
        for i, mats in tqdm(self._cache.items(), desc="Editing Layers"):
            layer_block = self.model.model.layers[i].mlp
            mod = {
                "gate": layer_block.gate_proj
                "up": layer_block.up_proj,
                "down": layer_block.down_proj
            }
            for name, m in mats.items():
                module = mod[name]
                with torch.no_grad():
                    A = mats[name][0]
                    G = mats[name][1]
                    tau = get_tau(module.weight, A, G)
                    tau = tau.to(self.device)
                    module.weight.add_(alpha * tau)
                    del A, G, tau
                torch.cuda.empty_cache()
        self._cache.clear()

    def save_model(self, save_path: str):
        print(f"Saving to {save_path} ---")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)