import sys
sys.path.insert(0, '.')

import argparse
import csv
import json
import math
import itertools
import os
import statistics

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader

from microsplit_framework import (
    ClientSpec, CleanAttack, ClusterJumpAttack,
    make_flat_sequential, MicrosplitModel, TopologySpec,
    DeviceBlueprint, client_specs_from_blueprint,
    inject_centroids,
)
from microsplit_framework.aggregation import MedianAggregation

from sweep_utils import build_devices, run_preprocessing


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    top1_correct = top5_correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
    return round(top1_correct / total * 100, 2), round(top5_correct / total * 100, 2)


def exhaustive_evaluate(
    base_model,
    devices,
    blueprint,
    m_attackers,
    centroids,
    eval_loader,
    get_layer_fn,
    set_layer_fn,
    device,
    aggregation=None,
) -> dict[frozenset, tuple[float, float]]:
    """
    Exhaustively evaluate all C(N, m) device-combo attack scenarios for a blueprint.

    Returns dict mapping frozenset[device_ids] → (top1, top5) for every combination.
    Uses the same swap-and-restore pattern as the inner loop — model is built once.
    """
    device_ids = [d.device_id for d in devices]
    client_specs = client_specs_from_blueprint(devices, blueprint, device_ids)
    n_total = len(device_ids)
    n_combos = math.comb(n_total, m_attackers)

    print(
        f"[Exhaustive] Blueprint: {blueprint}  |  m={m_attackers}  |  "
        f"N={n_total} devices  |  C(N,m)={n_combos} combos"
    )

    agg = aggregation or MedianAggregation()
    topology = TopologySpec(clients=[
        ClientSpec(
            client_id=cs.client_id,
            layer_range=cs.layer_range,
            height_range=cs.height_range,
            replicas=cs.replicas,
            aggregation=agg,
            replica_device_ids=cs.replica_device_ids,
        )
        for cs in client_specs
    ])
    model = MicrosplitModel(base_model, topology, get_layer_fn, set_layer_fn).to(device)
    inject_centroids(model, centroids)

    device_to_replicas: dict[str, list[tuple]] = {}
    for cm in model.client_modules:
        for i, dev_id in enumerate(cm.replica_device_ids):
            device_to_replicas.setdefault(dev_id, []).append((cm, i))

    _clean = CleanAttack()
    _malicious = ClusterJumpAttack()

    per_combo: dict[frozenset, tuple[float, float]] = {}
    for idx, combo in enumerate(itertools.combinations(device_ids, m_attackers), 1):
        for dev_id in combo:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _malicious

        top1, top5 = _evaluate(model, eval_loader, device)

        for dev_id in combo:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _clean

        combo_set = frozenset(combo)
        per_combo[combo_set] = (top1, top5)
        print(
            f"[Exhaustive]  [{idx}/{n_combos}] "
            f"Attackers: [{', '.join(sorted(combo))}]  →  Top-1: {top1:.2f}%"
        )

    del model

    top1_vals = [t1 for t1, _ in per_combo.values()]
    worst_combo = min(per_combo, key=lambda c: per_combo[c][0])
    worst_top1, worst_top5 = per_combo[worst_combo]
    mean_top1 = round(sum(top1_vals) / len(top1_vals), 2)
    median_top1 = round(statistics.median(top1_vals), 2)
    best_top1 = max(top1_vals)

    print(
        f"[Exhaustive] Done. C({n_total},{m_attackers})={n_combos} combos evaluated.\n"
        f"  Worst:  {{{', '.join(sorted(worst_combo))}}}  →  {worst_top1:.2f}%\n"
        f"  Median: {median_top1:.2f}%  |  Mean: {mean_top1:.2f}%  |  Best: {best_top1:.2f}%"
    )

    return per_combo


def write_csv(path: str, per_combo: dict[frozenset, tuple[float, float]]) -> None:
    top1_vals = [t1 for t1, _ in per_combo.values()]
    worst_combo = min(per_combo, key=lambda c: per_combo[c][0])
    mean_top1 = round(sum(top1_vals) / len(top1_vals), 2)
    median_top1 = round(statistics.median(top1_vals), 2)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attacker", "top1", "top5"])
        for combo, (t1, t5) in sorted(per_combo.items(), key=lambda x: x[1][0]):
            writer.writerow([" + ".join(sorted(combo)), f"{t1:.2f}", f"{t5:.2f}"])
        worst_t1, worst_t5 = per_combo[worst_combo]
        writer.writerow(["WORST",  f"{worst_t1:.2f}",   f"{worst_t5:.2f}"])
        writer.writerow(["MEAN",   f"{mean_top1:.2f}",  ""])
        writer.writerow(["MEDIAN", f"{median_top1:.2f}", ""])
    print(f"[Exhaustive] Results saved to {path}")


def _parse_blueprint(json_str: str) -> DeviceBlueprint:
    """
    Parse a blueprint from nested JSON:
        {"partition_id": {"device_id": count, ...}, ...}
    """
    raw = json.loads(json_str)
    d: dict[tuple[str, str], int] = {}
    for partition_id, dev_counts in raw.items():
        for device_id, count in dev_counts.items():
            d[(partition_id, device_id)] = int(count)
    return DeviceBlueprint.from_dict(d)


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustively evaluate all C(N,m) attack combos for a given blueprint.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--m", type=int, required=True,
        help="Number of simultaneously compromised devices",
    )
    parser.add_argument(
        "--blueprint", type=str, default=None,
        help=(
            "Blueprint as nested JSON string.\n"
            'Example: \'{"conv2": {"device_conv2": 2, "device_conv4": 1}}\'\n'
            "Omit for K=0 baseline (no extra replicas)."
        ),
    )
    parser.add_argument(
        "--blueprint-file", type=str, default=None,
        help="Path to a JSON file with the blueprint (same nested format as --blueprint).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: exhaustive_eval_m{m}.csv)",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=os.path.join("./imagenet_val_data", "imagenet_validation"),
        help="ImageNet validation directory",
    )
    args = parser.parse_args()

    if args.blueprint and args.blueprint_file:
        parser.error("Specify at most one of --blueprint or --blueprint-file.")

    if args.blueprint_file:
        with open(args.blueprint_file) as f:
            blueprint = _parse_blueprint(f.read())
    elif args.blueprint:
        blueprint = _parse_blueprint(args.blueprint)
    else:
        blueprint = DeviceBlueprint.from_dict({})

    out_path = args.out or f"exhaustive_eval_m{args.m}.csv"

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    weights = models.VGG16_Weights.DEFAULT
    print(f"Loading dataset from {args.data_dir}...")
    dataset = datasets.ImageFolder(root=args.data_dir, transform=weights.transforms())
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

    devices = build_devices()
    device_ids = [d.device_id for d in devices]

    _, _, final_centroids, final_eval_loader = run_preprocessing(
        flat_model, devices, device_ids, dataset, get_layer_fn, set_layer_fn, device,
    )

    per_combo = exhaustive_evaluate(
        base_model   = flat_model,
        devices      = devices,
        blueprint    = blueprint,
        m_attackers  = args.m,
        centroids    = final_centroids,
        eval_loader  = final_eval_loader,
        get_layer_fn = get_layer_fn,
        set_layer_fn = set_layer_fn,
        device       = device,
    )

    write_csv(out_path, per_combo)


if __name__ == "__main__":
    main()
