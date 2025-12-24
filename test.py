import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
from utils2 import *
from torch.utils.data import DataLoader
from curvature import *
from data import *
from einops import einsum, rearrange
from sklearn.decomposition import PCA

def process_layer_stats(model, tokenizer, dataset_name, layer_idx, num_batches, batch_size, cache_dir, seed):
    if check_layer_cache(cache_dir, dataset_name, layer_idx):
        return
    print(f"Generating stats for {dataset_name} - Layer {layer_idx}...")
    set_seed(seed)
    ds = SimpleTextDataset(f"data/{dataset_name}")
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
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=tokenize_collate)
    steering = Steering(model, tokenizer, layer_indices=[layer_idx], device="cuda")
    steering.collect_stats(loader, num_batches, dataset_name=dataset_name, cache_dir=cache_dir)
    del steering, loader, ds
    torch.cuda.empty_cache()
    gc.collect()

def KFAC_plotter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data1", type=str, default="toxic_train")
    parser.add_argument("--data2", type=str, default="nontoxic_train")
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="./stats_cache")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    args = validate_batch_size(args)
    set_seed(args.seed)

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, device="cuda")
    layers = list(range(model.config.num_hidden_layers))
    results = {"gate": [], "up": [], "down": []}
    rand_results = []
    print(f"Starting layer-wise comparison for {len(layers)} layers...")

    for i in layers:
        print(f"\n--- Processing Layer {i} ---")
        process_layer_stats(
            model, tokenizer, args.data1, i,
            args.num_batches, args.batch_size, args.cache_dir, args.seed
        )
        process_layer_stats(
            model, tokenizer, args.data2, i,
            args.num_batches, args.batch_size, args.cache_dir, args.seed
        )
        try:
            cache1 = load_layer_cache(args.cache_dir, args.data1, i)
            cache2 = load_layer_cache(args.cache_dir, args.data2, i)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        current_rand_vals = []
        for name in ["gate", "up", "down"]:
            A1, G1 = cache1[name]
            A2, G2 = cache2[name]
            results[name].append((fisher_overlap(A1, G1, A2, G2),
                                  (torch.trace(A1)*torch.trace(G1)/ A1.shape[0]*G1.shape[0]).item(),
                                  (torch.trace(A2)*torch.trace(G2)/A1.shape[0]*G1.shape[0]).item()))

            r_sim = fisher_overlap(
                torch.randn_like(A1), torch.randn_like(G1),
                torch.randn_like(A2), torch.randn_like(G2)
            )
            current_rand_vals.append(r_sim)
        rand_results.append(np.mean(current_rand_vals))
        del cache1, cache2, A1, G1, A2, G2
        gc.collect()

    # Plot similarity
    gate_data = np.array(results["gate"])
    up_data = np.array(results["up"])
    down_data = np.array(results["down"])

    # Plot similarity
    plt.figure(figsize=(10, 6))
    plt.plot(layers, rand_results, label="random", linestyle="--", color="gray")

    # Use slicing [:, 0] to get the 0th element for ALL layers
    plt.plot(layers, gate_data[:, 0], label="gate", color='blue')
    plt.plot(layers, up_data[:, 0], label="up", color='red')
    plt.plot(layers, down_data[:, 0], label="down", color='green')

    plt.xlabel("Layer Index")
    plt.ylabel("Fisher cosine similarity")
    plt.legend()
    #plt.grid(False, alpha=0.3)
    plt.savefig("similarity_plot.png")
    plt.close()

    # Plot magnitude ratio (Column 1 / Column 2)
    plt.figure(figsize=(10, 6))

    # We divide the arrays element-wise: Trace(D1) / Trace(D2)
    plt.plot(layers, gate_data[:, 1] , label="gate(toxic)", color='blue')
    plt.plot(layers, up_data[:, 1], label="up(toxic)", color='red')
    plt.plot(layers, down_data[:, 1], label="down (toxic)", color='green')
    plt.plot(layers, gate_data[:, 2], label="gate(nontoxic)", color='blue', linestyle="--")
    plt.plot(layers, up_data[:, 2], label="up(nontoxic)", color='red', linestyle="--")
    plt.plot(layers, down_data[:, 2], label="down (nontoxic)", color='green', linestyle="--")
    plt.xlabel("Layer Index")
    plt.ylabel("Trace Magnitude")
    plt.legend()
    #plt.grid(False, alpha=0.3)
    plt.yscale('log')
    plt.savefig("magnitude_plot.png")
    plt.close()


def weight_plotter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data1", type=str, default="toxic_train")
    parser.add_argument("--data2", type=str, default="nontoxic_train")
    parser.add_argument("--layers", nargs='+', type=int, required=True)
    parser.add_argument("--cache_dir", type=str, default="./stats_cache")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--weight_proj", type=str, default="up")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("graphs", exist_ok=True)
    print(f"Loading model {args.model_path}...")
    model, tokenizer = load_model(args.model_path, device="cuda")
    num_plots = len(args.layers)
    fig_def, axs_def = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
    fig_comp, axs_comp = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
    fig_all, axs_all = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
    axs_def = axs_def.flatten()
    axs_comp = axs_comp.flatten()
    axs_all = axs_all.flatten()

    for idx, i in enumerate(args.layers):
        print(f"Processing Layer {i}...")
        process_layer_stats(model, tokenizer, args.data1, i, 25, 1, args.cache_dir, args.seed)
        process_layer_stats(model, tokenizer, args.data2, i, 25, 1, args.cache_dir, args.seed)
        try:
            cache1 = load_layer_cache(args.cache_dir, args.data1, i)
            cache2 = load_layer_cache(args.cache_dir, args.data2, i)
        except FileNotFoundError as e:
            print(f"Error loading layer {i}: {e}")
            continue

        layer_mlp = model.model.layers[i].mlp
        if args.weight_proj == "gate":
            mod = layer_mlp.gate_proj
        elif args.weight_proj == "up":
            mod = layer_mlp.up_proj
        elif args.weight_proj == "down":
            mod = layer_mlp.down_proj
        else:
            raise ValueError(f"Unknown weight projection: {args.weight_proj}")
        w = mod.weight.detach().cpu().to(torch.float)

        A1, G1 = cache1[args.weight_proj]
        A2, G2 = cache2[args.weight_proj]
        A1, G1 = A1.float(), G1.float()
        A2, G2 = A2.float(), G2.float()
        w_d1_kfac = args.alpha * get_tau(w, A1, G1)
        w_d2_kfac = args.alpha * get_tau(w, A2, G2)
        #Activation Projection
        w_act = einsum(w, A1, "d_out i, i d_in -> d_out d_in")
        w_act =args.alpha * w_act / (w_act.norm() + 1e-8)
        w_grad = einsum(G1, w,"d_out i, i d_in -> d_out d_in")
        w_grad = args.alpha * w_grad / (w_grad.norm() + 1e-8)

        if args.weight_proj == "down":
            w_d1_kfac = rearrange(w_d1_kfac , "row col -> col row")
            w_d2_kfac = rearrange(w_d2_kfac , "row col -> col row")
            w_act = rearrange(w_act, "row col -> col row")
            w_grad = rearrange(w_grad, "row col -> col row")

        def plot_on_axis(ax, tensors, labels, markers, colors, title):
            data = [t.detach().cpu().numpy() for t in tensors]
            combined = np.vstack(data)

            # PCA
            pca = PCA(n_components=2)
            proj = pca.fit_transform(combined)

            start = 0
            for k, d in enumerate(data):
                end = start + d.shape[0]
                ax.scatter(
                    proj[start:end, 0],
                    proj[start:end, 1],
                    label=labels[k],
                    marker=markers[k],
                    c=colors[k],
                    alpha=0.6,
                    s=15,
                    edgecolors='white',  # Adds separation if dots overlap
                    linewidth=0.3
                )
                start = end

            ax.set_title(title)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            if idx == 0:  # Only legend on the first plot to avoid clutter
                ax.legend(loc='upper right', fontsize='small')

        # 1. Fill Plot 1 (KFAC)
        plot_on_axis(
            axs_def[idx],
            [w_d1_kfac, w_d2_kfac],
            [f"{args.data1} K-FAC", f"{args.data2} F-FAC"],
            ['o', 'X'],
            ['red', 'green'],
            f"Layer {i}"
        )

        # 2. Fill Plot 2 (Activation Comp)
        plot_on_axis(
            axs_comp[idx],
            [w_d1_kfac, w_act],
            [f"{args.data1} K-FAC", f"{args.data1} Act Proj"],
            ['o', 'v'],
            ['red', 'blue'],
            f"Layer {i}"
        )

        plot_on_axis(
            axs_all[idx],
            [w_d1_kfac, w_grad],
            [f"{args.data1} K-FAC", f"{args.data1} Grad Proj"],
            ['o', '^'],
            ['red', 'green'],
            f"Layer {i}"
        )

    print("Saving plots...")

    fig_def.suptitle(f"Toxic vs nontoxic task detection via K-FAC ({args.weight_proj})")
    fig_def.tight_layout()
    fig_def.savefig("graphs/all_layers_kfac_2.png", dpi=300)
    plt.close(fig_def)

    fig_comp.suptitle(f"Toxic task detection K-FAC vs Activation Projection ({args.weight_proj})")
    fig_comp.tight_layout()
    fig_comp.savefig("graphs/all_layers_act_2.png", dpi=300)
    plt.close(fig_comp)

    fig_all.suptitle(f"Toxic task detection K-FAC vs Grad Projection  ({args.weight_proj})")
    fig_all.tight_layout()
    fig_all.savefig("graphs/all_layers_combined_2.png", dpi=300)
    plt.close(fig_all)

    print("Done. Graphs saved to ./graphs/")

if __name__ == "__main__":
    #KFAC_plotter()
    weight_plotter()