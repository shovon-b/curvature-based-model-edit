import argparse
import os
import gc
import torch
from torch.utils.data import DataLoader
from utils2 import *
from curvature import Steering
from data import *

def editor_function():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_base_dir", type=str, default="./steered_models_2")
    parser.add_argument("--data", type=str, default="math_sycophantic")
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.0, help="Steering strength")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cache_dir", type=str, default="./stats_cache")
    args = parser.parse_args()
    args = validate_batch_size(args)
    set_seed(args.seed)

    base_model_name = args.model_path.replace("/", "-")
    save_dir_name = f"{base_model_name}_{args.data[:3]}_alpha{args.alpha}_gate"
    save_path = os.path.join(args.output_base_dir, save_dir_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)

    def tokenize_collate(batch):
        encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        labels = encoded.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        return {
            'input_ids': encoded.input_ids,
            'attention_mask': encoded.attention_mask,
            'labels': labels
        }

    num_layers = model.config.num_hidden_layers
    target_layers = list(range(num_layers//2 - 8, num_layers//2 + 4))
    print(f"Editing layers: {target_layers}")

    editor = Steering(
        model,
        tokenizer,
        layer_indices=target_layers,
        device=model.device
    )
    if args.alpha != 0.0:
        print(f"Alpha ({args.alpha}). Collecting stats for data {args.data}")
        ds = SimpleTextDataset(f"data/{args.data}")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=tokenize_collate)

        editor.collect_stats(
            dataloader=loader,
            num_batches=args.num_batches,
            dataset_name=args.data,
            cache_dir=args.cache_dir
        )
    else:
        print("Alpha is 0.0. Skipping collection.")

    print("Editing and Saving model")
    editor.edit_layers(alpha=args.alpha)
    editor.save_model(save_path=save_path)

if __name__ == "__main__":
    editor_function()