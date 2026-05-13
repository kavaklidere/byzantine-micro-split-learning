import sys
sys.path.insert(0, '.')

import os
import csv
import random as _random
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from microsplit_framework import (
    ClientSpec, CleanAttack, make_flat_sequential,
    MicrosplitModel, TopologySpec,
    preprocess_clustering,
    get_eval_indices, make_balanced_mini_loader, extract_centroids,
    RedundancyBlueprint, evaluate_inner_loop,
)
from microsplit_framework.ga_outer_loop import GAConfig, GAResult, run_ga


# ── Sweep parameters ──────────────────────────────────────────────────────────
BUDGETS     = [1, 2, 3, 4, 5]   # K: extra replicas distributed across clients
M_ATTACKERS = [1]                # m: simultaneous malicious replicas per evaluation

# ── GA hyperparameters (shared across all runs) ────────────────────────────────
POPULATION_SIZE  = 25
GENERATIONS      = 15
N_SEED_FROM_PREV = 1   # how many initial population slots come from the previous budget's best blueprint

# ── Preprocessing hyperparameters ─────────────────────────────────────────────
N_PREPROCESS_BATCHES = 150
BATCH_SIZE           = 64
N_CLUSTERS           = 100
N_PER_CLASS          = 1


def _expand_blueprint(bp, client_ids):
    """All blueprints reachable by adding one replica to any client of bp."""
    seeds = []
    for cid in client_ids:
        extras = {c: bp.n_replicas(c) - 1 for c in client_ids}
        extras[cid] += 1
        seeds.append(RedundancyBlueprint.from_dict({c: e for c, e in extras.items() if e > 0}))
    return seeds


def build_client_specs():
    return [
        # Conv clients
        ClientSpec("conv1_top",   layer_range=(0,   4),  height_range=(0.0, 0.5), replicas=[CleanAttack()]),
        ClientSpec("conv1_bot",   layer_range=(0,   4),  height_range=(0.5, 1.0), replicas=[CleanAttack()]),
        ClientSpec("conv2",       layer_range=(5,   9),  height_range=(0.0, 1.0), replicas=[CleanAttack()]),
        ClientSpec("conv3_top",   layer_range=(10, 16),  height_range=(0.0, 0.5), replicas=[CleanAttack()]),
        ClientSpec("conv3_bot",   layer_range=(10, 16),  height_range=(0.5, 1.0), replicas=[CleanAttack()]),
        ClientSpec("conv4",       layer_range=(17, 23),  height_range=(0.0, 1.0), replicas=[CleanAttack()]),
        # conv5 absorbs avgpool (index 31)
        ClientSpec("conv5_top",   layer_range=(24, 31),  height_range=(0.0, 0.5), replicas=[CleanAttack()]),
        ClientSpec("conv5_bot",   layer_range=(24, 31),  height_range=(0.5, 1.0), replicas=[CleanAttack()]),
        # Dense clients — block 1: Flatten + fc1 (32–35), block 2: fc2 + fc3 (36–39)
        ClientSpec("dense1_top",  layer_range=(32, 35),  height_range=(0.0, 0.5), replicas=[CleanAttack()]),
        ClientSpec("dense1_bot",  layer_range=(32, 35),  height_range=(0.5, 1.0), replicas=[CleanAttack()]),
        ClientSpec("dense2_top",  layer_range=(36, 39),  height_range=(0.0, 0.5), replicas=[CleanAttack()]),
        ClientSpec("dense2_bot",  layer_range=(36, 39),  height_range=(0.5, 1.0), replicas=[CleanAttack()]),
    ]


def run_preprocessing(flat_model, base_client_specs, dataset, get_layer_fn, set_layer_fn, device):
    print("=== Preprocessing (runs once, shared across all sweep experiments) ===")

    eval_idx_set = set(get_eval_indices(dataset, N_PER_CLASS, seed=42))
    pre_pool = [i for i in range(len(dataset)) if i not in eval_idx_set]
    _random.Random(42).shuffle(pre_pool)
    n_pre = N_PREPROCESS_BATCHES * BATCH_SIZE

    _g = torch.Generator()
    _g.manual_seed(42)
    pre_loader = DataLoader(
        Subset(dataset, pre_pool[:n_pre]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, generator=_g,
    )
    mini_eval_loader = make_balanced_mini_loader(dataset, N_PER_CLASS, BATCH_SIZE, seed=42)
    print(f"Mini-eval: {len(mini_eval_loader.dataset)} images, no overlap with preprocessing pool\n")

    preprocess_topology = TopologySpec(clients=[
        ClientSpec(
            client_id=spec.client_id, layer_range=spec.layer_range,
            height_range=spec.height_range, replicas=[CleanAttack()],
        )
        for spec in base_client_specs
    ])
    preprocess_model = MicrosplitModel(
        flat_model, preprocess_topology, get_layer_fn, set_layer_fn,
    ).to(device)

    preprocess_clustering(
        preprocess_model, pre_loader,
        n_batches=N_PREPROCESS_BATCHES, n_clusters=N_CLUSTERS, device=device,
    )
    centroids = extract_centroids(preprocess_model)
    del preprocess_model
    print(f"Centroids ready for: {list(centroids.keys())}\n")
    return centroids, mini_eval_loader


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    weights = models.VGG16_Weights.DEFAULT
    data_dir = os.path.join('./imagenet_val_data', 'imagenet_validation')
    print(f"Loading dataset from {data_dir}...")
    dataset = datasets.ImageFolder(root=data_dir, transform=weights.transforms())
    print(f"Dataset: {len(dataset)} images\n")

    base_model = models.vgg16(weights=weights)
    flat_model = make_flat_sequential(
        base_model.features,
        base_model.avgpool,
        nn.Flatten(),
        base_model.classifier,
    )
    get_layer_fn = lambda m, i: m[i]
    set_layer_fn = lambda m, i, v: m.__setitem__(i, v)

    base_client_specs = build_client_specs()

    # Preprocessing — once for all (K, M) combinations
    centroids, mini_eval_loader = run_preprocessing(
        flat_model, base_client_specs, dataset, get_layer_fn, set_layer_fn, device,
    )

    os.makedirs("results/ga_sweep", exist_ok=True)

    summary_worst: dict[int, dict[int, float]] = {k: {} for k in [0] + BUDGETS}
    summary_avg:   dict[int, dict[int, float]] = {k: {} for k in [0] + BUDGETS}

    for m in M_ATTACKERS:
        # ── K=0 baseline for this m: single inner-loop call, no GA needed ─────
        print(f"\n{'='*60}")
        print(f"  Baseline — K=0 (no redundancy), m_attackers={m}")
        print(f"{'='*60}")
        baseline_bp = RedundancyBlueprint.from_dict({})
        baseline_result = evaluate_inner_loop(
            base_model        = flat_model,
            base_client_specs = base_client_specs,
            blueprint         = baseline_bp,
            m_attackers       = m,
            centroids         = centroids,
            mini_eval_loader  = mini_eval_loader,
            get_layer_fn      = get_layer_fn,
            set_layer_fn      = set_layer_fn,
            device            = device,
            verbose           = True,
        )
        baseline_avg_top1 = round(sum(t1 for t1, _ in baseline_result.per_combo.values()) / len(baseline_result.per_combo), 2)
        worst_str = ", ".join(f"{cid}[{i}]" for cid, i in sorted(baseline_result.worst_combo))
        print(f"\nBaseline Worst-Case Top-1: {baseline_result.min_top1:.2f}%")
        print(f"Baseline Worst-Case Top-5: {baseline_result.min_top5:.2f}%")
        print(f"Baseline Avg-Case Top-1  : {baseline_avg_top1:.2f}%")
        print(f"Worst Attacker           : [{worst_str}]")
        summary_worst[0][m] = baseline_result.min_top1
        summary_avg[0][m]   = baseline_avg_top1

        baseline_path = f"results/ga_sweep/ga_k0_m{m}_baseline.csv"
        with open(baseline_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["attacker", "top1", "top5"])
            for combo, (t1, t5) in sorted(baseline_result.per_combo.items(), key=lambda x: x[1][0]):
                combo_str = " + ".join(f"{cid}[{i}]" for cid, i in sorted(combo))
                writer.writerow([combo_str, f"{t1:.2f}", f"{t5:.2f}"])
            writer.writerow(["AVERAGE", f"{baseline_avg_top1:.2f}", ""])
        print(f"Baseline results saved to {baseline_path}")

        prev_best_blueprint: RedundancyBlueprint | None = None
        client_ids = [spec.client_id for spec in base_client_specs]

        for k in BUDGETS:
            print(f"\n{'='*60}")
            print(f"  GA run — budget K={k}, m_attackers={m}")
            print(f"{'='*60}")

            seed_blueprints = (
                _expand_blueprint(prev_best_blueprint, client_ids)
                if prev_best_blueprint is not None else None
            )

            config = GAConfig(
                budget          = k,
                population_size = POPULATION_SIZE,
                generations     = GENERATIONS,
                m_attackers     = m,
                n_seed_from_prev = N_SEED_FROM_PREV,
            )

            result: GAResult = run_ga(
                base_model        = flat_model,
                base_client_specs = base_client_specs,
                dataset           = dataset,
                config            = config,
                get_layer_fn      = get_layer_fn,
                set_layer_fn      = set_layer_fn,
                device            = device,
                verbose           = False,
                centroids         = centroids,
                mini_eval_loader  = mini_eval_loader,
                seed              = 42,
                seed_blueprints   = seed_blueprints,
            )
            prev_best_blueprint = result.best_blueprint

            worst_str = ", ".join(f"{cid}[{i}]" for cid, i in sorted(result.best_worst_combo))
            print(f"\nBest Blueprint  : {result.best_blueprint}")
            print(f"Worst-Case Top-1: {result.best_min_top1:.2f}%")
            print(f"Worst-Case Top-5: {result.best_min_top5:.2f}%")
            print(f"Avg-Case Top-1  : {result.best_bp_avg_top1:.2f}%")
            print(f"Worst Attacker  : [{worst_str}]")

            summary_worst[k][m] = result.best_min_top1
            summary_avg[k][m]   = result.best_bp_avg_top1

            # Per-experiment history CSV
            hist_path = f"results/ga_sweep/ga_k{k}_m{m}_history.csv"
            with open(hist_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["generation", "best_top1", "best_bp_avg_top1", "pop_mean_top1", "best_blueprint", "worst_attacker"]
                )
                writer.writeheader()
                writer.writerows(result.generation_history)
            print(f"History saved to {hist_path}")

    # Summary CSVs — rows: K (budget, 0=baseline), columns: M (m_attackers)
    for label, data in [("worst", summary_worst), ("avg", summary_avg)]:
        path = f"results/ga_sweep/summary_{label}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["budget \\ m_attackers"] + [str(m) for m in M_ATTACKERS])
            for k in [0] + BUDGETS:
                writer.writerow([k] + [f"{data[k][m]:.2f}" for m in M_ATTACKERS])
        print(f"\nSummary ({label}) saved to {path}")


if __name__ == "__main__":
    main()
