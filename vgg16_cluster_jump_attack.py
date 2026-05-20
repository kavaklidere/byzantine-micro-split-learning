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
    ClientSpec, CleanAttack, ClusterJumpAttack,
    make_flat_sequential, MicrosplitModel, TopologySpec,
    preprocess_clustering,
)

SEED = 42

# ── Target positions in the VGG16 flat model ──────────────────────────────────
POSITIONS = {
    "conv2":  dict(block=(5,  9),  before=(0,  4),  after=(10, 39)),
    "conv4":  dict(block=(17, 23), before=(0, 16),  after=(24, 39)),
    "dense1": dict(block=(32, 35), before=(0, 31),  after=(36, 39)),
}

# ── Fraction of the block height the malicious client controls ─────────────────
FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# ── Preprocessing hyperparameters ─────────────────────────────────────────────
N_PREPROCESS_BATCHES = 150
BATCH_SIZE           = 64
N_CLUSTERS           = 100


def build_client_specs(block, before, after, fraction):
    """
    Returns a list of ClientSpecs for one experiment.
    5 clients for fraction < 1.0 (before / above / malicious / below / after).
    3 clients for fraction = 1.0 (before / malicious / after) — no room for above/below.
    """
    specs = [ClientSpec("before", layer_range=before, height_range=(0.0, 1.0), replicas=[CleanAttack()])]

    if fraction < 1.0:
        h_low  = (1.0 - fraction) / 2
        h_high = (1.0 + fraction) / 2
        specs += [
            ClientSpec("above",    layer_range=block, height_range=(0.0,   h_low),  replicas=[CleanAttack()]),
            ClientSpec("malicious",layer_range=block, height_range=(h_low, h_high), replicas=[ClusterJumpAttack()]),
            ClientSpec("below",    layer_range=block, height_range=(h_high, 1.0),   replicas=[CleanAttack()]),
        ]
    else:
        specs.append(
            ClientSpec("malicious", layer_range=block, height_range=(0.0, 1.0), replicas=[ClusterJumpAttack()])
        )

    specs.append(ClientSpec("after", layer_range=after, height_range=(0.0, 1.0), replicas=[CleanAttack()]))
    return specs


def run_experiment(flat_model, client_specs, dataset, get_layer_fn, set_layer_fn, device):
    """
    Shuffle the dataset once, use the first N_PRE images for KMeans warmup,
    evaluate on the remaining images (no overlap).
    ClusterJumpAttack is dormant during preprocessing (is_cluster_ready=False) and
    activates automatically once preprocessing sets is_cluster_ready=True.
    """
    topology = TopologySpec(clients=client_specs)
    model    = MicrosplitModel(flat_model, topology, get_layer_fn, set_layer_fn).to(device)

    # Single shuffle — first slice for preprocessing, remainder for evaluation
    pool  = list(range(len(dataset)))
    _random.Random(SEED).shuffle(pool)
    n_pre = N_PREPROCESS_BATCHES * BATCH_SIZE
    pre_loader = DataLoader(
        Subset(dataset, pool[:n_pre]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    preprocess_clustering(model, pre_loader, n_batches=N_PREPROCESS_BATCHES, n_clusters=N_CLUSTERS, device=device)

    # Eval DataLoader — remaining (non-preprocessing) images, already in shuffled order
    eval_loader = DataLoader(
        Subset(dataset, pool[n_pre:]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    model.eval()
    top1_correct = top5_correct = total = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct    = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)

    del model
    return round(top1_correct / total * 100, 2), round(top5_correct / total * 100, 2)


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    weights  = models.VGG16_Weights.DEFAULT
    data_dir = os.path.join('./imagenet_val_data', 'imagenet_validation')
    print(f"Loading dataset from {data_dir}...")
    dataset  = datasets.ImageFolder(root=data_dir, transform=weights.transforms())
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

    os.makedirs("results/vgg16/cluster_jump_attack", exist_ok=True)

    for pos_name, pos in POSITIONS.items():
        print(f"\n{'='*60}")
        print(f"  Position: {pos_name}  (layers {pos['block'][0]}–{pos['block'][1]})")
        print(f"{'='*60}")

        top1_rows = []
        top5_rows = []

        for fraction in FRACTIONS:
            frac_pct  = int(fraction * 100)
            n_clients = 3 if fraction == 1.0 else 5
            print(f"\n  fraction={frac_pct}%  ({n_clients} clients) ...")

            specs        = build_client_specs(pos["block"], pos["before"], pos["after"], fraction)
            top1, top5   = run_experiment(flat_model, specs, dataset, get_layer_fn, set_layer_fn, device)

            print(f"  Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")
            top1_rows.append({"fraction_pct": frac_pct, "top1": top1})
            top5_rows.append({"fraction_pct": frac_pct, "top5": top5})

        for label, rows, col in [("top1", top1_rows, "top1"), ("top5", top5_rows, "top5")]:
            path = f"results/vgg16/cluster_jump_attack/{pos_name}_{label}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["fraction_pct", col])
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
