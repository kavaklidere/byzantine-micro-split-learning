from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microsplit_framework.topology import ClientSpec, TopologySpec, DeviceSpec
from microsplit_framework.attacks import CleanAttack, ClusterJumpAttack
from microsplit_framework.aggregation import AggregationStrategy, MedianAggregation
from microsplit_framework.model import MicrosplitModel

# Re-export unchanged types and helpers from v1
from microsplit_framework.ga_inner_loop_v1 import (
    RedundancyBlueprint,
    DeviceBlueprint,
    client_specs_from_blueprint,
    extract_centroids,
    inject_centroids,
    get_eval_indices,
    make_balanced_mini_loader,
)


# ── Result types ────────────────────────────────────────────────────────────────

@dataclass
class InnerLoopResult:
    blueprint:   DeviceBlueprint   # was annotated RedundancyBlueprint in v1 (bug fix)
    m_attackers: int
    min_top1:    float
    min_top5:    float
    worst_combo: frozenset
    per_combo:   dict = field(default_factory=dict)   # only the evaluated T candidates


@dataclass
class AttackPotentials:
    clean_acc:    float
    potentials:   dict[str, float]                  # p_σ = clean_acc - acc(σ attacked alone)
    interactions: dict[tuple[str, str], float]      # Δ_{σ1,σ2} (key always sorted)
    path_groups:  list[frozenset[str]]              # overlap groups for suspicious-set injection


# ── Private helpers ─────────────────────────────────────────────────────────────

def _evaluate_mini_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
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


def _compute_path_groups(client_specs: list[ClientSpec]) -> list[frozenset[str]]:
    """
    Union-find grouping: two regions share a group iff their height_range intervals
    strictly overlap (a[0] < b[1] and b[0] < a[1]).
    Groups of size 1 are omitted — a lone region has no spatially coherent partner.
    """
    region_ids = [cs.client_id for cs in client_specs]
    hr = {cs.client_id: cs.height_range for cs in client_specs}

    parent = {r: r for r in region_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for i, r1 in enumerate(region_ids):
        for r2 in region_ids[i + 1:]:
            a, b = hr[r1], hr[r2]
            if a[0] < b[1] and b[0] < a[1]:
                union(r1, r2)

    groups: dict[str, set[str]] = {}
    for r in region_ids:
        groups.setdefault(find(r), set()).add(r)

    return [frozenset(g) for g in groups.values() if len(g) > 1]


# ── Surrogate ranking helper ────────────────────────────────────────────────────

def compute_surrogate_ranking(
    device_ids: list[str],
    m_attackers: int,
    client_specs: list[ClientSpec],
    potentials: "AttackPotentials",
) -> tuple[list[frozenset], dict[frozenset, float]]:
    """
    Score all C(N,m) device combos by surrogate D̂ and return them ranked descending.
    Returns (sorted_combos, scores): sorted_combos[0] is the highest-D̂ combo.
    Shared by the production inner loop and the brute/top-K comparison script.
    """
    sigma_device_counts: dict[str, dict[str, int]] = {}
    r_sigma: dict[str, int] = {}
    for cs in client_specs:
        counts: dict[str, int] = {}
        for dev_id in cs.replica_device_ids:
            counts[dev_id] = counts.get(dev_id, 0) + 1
        sigma_device_counts[cs.client_id] = counts
        r_sigma[cs.client_id] = len(cs.replica_device_ids)
    t_sigma = {sigma: math.ceil(r / 2) for sigma, r in r_sigma.items()}
    all_regions = list(sigma_device_counts.keys())

    p = potentials.potentials
    delta = potentials.interactions

    def _corruptible(combo: tuple[str, ...]) -> frozenset[str]:
        S = set(combo)
        return frozenset(
            sigma for sigma in all_regions
            if sum(sigma_device_counts[sigma].get(d, 0) for d in S) >= t_sigma[sigma]
        )

    def _surrogate_score(corr: frozenset[str]) -> float:
        score = sum(p.get(sigma, 0.0) for sigma in corr)
        corr_list = sorted(corr)
        for i, s1 in enumerate(corr_list):
            for s2 in corr_list[i + 1:]:
                score += delta.get(tuple(sorted([s1, s2])), 0.0)
        return score

    all_scores: dict[frozenset, float] = {}
    for combo in itertools.combinations(device_ids, m_attackers):
        corr = _corruptible(combo)
        all_scores[frozenset(combo)] = _surrogate_score(corr)

    sorted_combos = sorted(all_scores, key=all_scores.__getitem__, reverse=True)
    return sorted_combos, all_scores


# ── Stage 0: compute_attack_potentials ─────────────────────────────────────────

def compute_attack_potentials(
    base_model: nn.Module,
    devices: list[DeviceSpec],
    centroids: dict[str, torch.Tensor | None],
    mini_eval_loader: DataLoader,
    get_layer_fn: Callable,
    set_layer_fn: Callable | None,
    device: torch.device,
) -> AttackPotentials:
    """
    Compute per-region attack potentials (p_σ) and pairwise interactions (Δ_{σ1,σ2})
    on the K=0 topology (1 replica per region, full attacker control).

    Total inference calls: 1 (clean) + N (singles) + N(N-1)/2 (pairs).
    All inference on mini_eval_loader.
    """
    device_ids = [d.device_id for d in devices]

    baseline_bp = DeviceBlueprint.from_dict({})
    client_specs = client_specs_from_blueprint(devices, baseline_bp, device_ids)
    all_regions = [cs.client_id for cs in client_specs]

    N = len(all_regions)
    n_pairs = N * (N - 1) // 2
    total_calls = 1 + N + n_pairs
    print(
        f"[AttackPotentials] K=0 topology: {N} regions, {n_pairs} pairs "
        f"→ {total_calls} inference calls total"
    )

    agg = MedianAggregation()
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

    region_to_cm = {cm.client_id: cm for cm in model.client_modules}

    _clean = CleanAttack()
    _malicious = ClusterJumpAttack()

    # Baseline (no attack)
    clean_top1, _ = _evaluate_mini_batch(model, mini_eval_loader, device)
    clean_acc = clean_top1
    print(f"[AttackPotentials] Clean baseline: {clean_acc:.2f}%")

    # Per-region potentials
    print(f"[AttackPotentials] Potentials:")
    potentials_dict: dict[str, float] = {}
    for i, sigma in enumerate(all_regions, 1):
        cm = region_to_cm[sigma]
        cm.attack_configs[0] = _malicious
        acc_sigma, _ = _evaluate_mini_batch(model, mini_eval_loader, device)
        cm.attack_configs[0] = _clean
        p_sigma = clean_acc - acc_sigma
        potentials_dict[sigma] = p_sigma
        print(f"  [{i}/{N}] {sigma:<35} attack alone → p = {p_sigma:+.2f}%")

    # Pairwise interactions
    print(f"[AttackPotentials] Pairs ({n_pairs} calls):")
    interactions: dict[tuple[str, str], float] = {}
    pair_idx = 0
    for i, sigma1 in enumerate(all_regions):
        for sigma2 in all_regions[i + 1:]:
            pair_idx += 1
            cm1 = region_to_cm[sigma1]
            cm2 = region_to_cm[sigma2]
            cm1.attack_configs[0] = _malicious
            cm2.attack_configs[0] = _malicious
            acc_pair, _ = _evaluate_mini_batch(model, mini_eval_loader, device)
            cm1.attack_configs[0] = _clean
            cm2.attack_configs[0] = _clean
            D_pair = clean_acc - acc_pair
            delta = D_pair - potentials_dict[sigma1] - potentials_dict[sigma2]
            key = tuple(sorted([sigma1, sigma2]))
            interactions[key] = delta
            print(
                f"  [{pair_idx}/{n_pairs}] {sigma1} × {sigma2}"
                f"  →  D = {D_pair:.2f}%  Δ = {delta:+.2f}%"
            )

    # Summary
    max_sigma = max(potentials_dict, key=potentials_dict.get)
    delta_vals = list(interactions.values())
    n_neg = sum(1 for d in delta_vals if d < 0)
    n_pos = len(delta_vals) - n_neg
    print(
        f"[AttackPotentials] Done. Max p_σ = {potentials_dict[max_sigma]:.2f}% ({max_sigma}).\n"
        f"  Pair range: Δ ∈ [{min(delta_vals):.1f}%, {max(delta_vals):.1f}%]"
        f"  ({n_neg} negative, {n_pos} non-negative)"
    )

    path_groups = _compute_path_groups(client_specs)

    del model

    return AttackPotentials(
        clean_acc=clean_acc,
        potentials=potentials_dict,
        interactions=interactions,
        path_groups=path_groups,
    )


# ── Main inner loop ─────────────────────────────────────────────────────────────

def evaluate_inner_loop(
    base_model: nn.Module,
    devices: list[DeviceSpec],
    blueprint: DeviceBlueprint,
    m_attackers: int,
    centroids: dict[str, torch.Tensor | None],
    eval_loader: DataLoader,
    get_layer_fn: Callable,
    set_layer_fn: Callable | None,
    device: torch.device,
    potentials: AttackPotentials,
    T_provisional: int = 15,
    aggregation: AggregationStrategy | None = None,
    verbose: bool = False,
) -> InnerLoopResult:
    """
    Surrogate-guided inner loop: scores all C(N,m) device combos arithmetically,
    evaluates only the top-T_provisional with real inference.

    For C(N,m) ≤ T_provisional the evaluation is exhaustive (same as v1).
    The eval_loader can be mini_eval_loader (during GA) or final_eval_loader
    (post-GA held-out check in the sweep script) — the function is agnostic.
    """
    if potentials is None:
        raise NotImplementedError(
            "potentials=None is not supported. "
            "Call compute_attack_potentials() before the GA and pass the result here. "
            "For the old exhaustive path, use ga_inner_loop_v1.evaluate_inner_loop."
        )

    device_ids = [d.device_id for d in devices]
    client_specs = client_specs_from_blueprint(devices, blueprint, device_ids)

    n_total = len(device_ids)
    n_combos = math.comb(n_total, m_attackers)
    print(
        f"[InnerLoop] Blueprint: {blueprint}  |  m={m_attackers}  |  "
        f"N={n_total} devices  |  C(N,m)={n_combos} combos"
    )

    agg = aggregation or MedianAggregation()

    # ── Surrogate ranking ────────────────────────────────────────────────────────
    sorted_combos, all_scores = compute_surrogate_ranking(
        device_ids, m_attackers, client_specs, potentials
    )
    top_combos = sorted_combos[:T_provisional]

    top_score    = all_scores[top_combos[0]]  if top_combos else 0.0
    bottom_score = all_scores[top_combos[-1]] if top_combos else 0.0
    top_str      = ", ".join(sorted(top_combos[0])) if top_combos else ""
    print(
        f"[InnerLoop] Surrogate: scored {n_combos} combos "
        f"→ top-{len(top_combos)} candidates selected\n"
        f"            Top surrogate: {{{top_str}}} D̂={top_score:.2f}  "
        f"|  Bottom candidate D̂={bottom_score:.2f}"
    )

    # Path-group injections: one per group, targeting devices with most replica coverage
    sigma_device_counts: dict[str, dict[str, int]] = {}
    for cs in client_specs:
        counts: dict[str, int] = {}
        for dev_id in cs.replica_device_ids:
            counts[dev_id] = counts.get(dev_id, 0) + 1
        sigma_device_counts[cs.client_id] = counts

    candidate_pool: list[frozenset] = list(top_combos)
    candidate_set: set[frozenset] = set(candidate_pool)

    for group in potentials.path_groups:
        device_group_scores: dict[str, int] = {}
        for sigma in group:
            if sigma in sigma_device_counts:
                for dev_id, count in sigma_device_counts[sigma].items():
                    device_group_scores[dev_id] = device_group_scores.get(dev_id, 0) + count
        if len(device_group_scores) >= m_attackers:
            top_devices = sorted(
                device_group_scores, key=device_group_scores.__getitem__, reverse=True
            )[:m_attackers]
            injection = frozenset(top_devices)
            if injection not in candidate_set:
                candidate_pool.append(injection)
                candidate_set.add(injection)

    # ── Real inference on candidate pool ────────────────────────────────────────
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
    min_top1 = float("inf")
    min_top5 = float("inf")
    worst_combo: frozenset = frozenset()

    for idx, combo_set in enumerate(candidate_pool, 1):
        for dev_id in combo_set:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _malicious

        top1, top5 = _evaluate_mini_batch(model, eval_loader, device)

        for dev_id in combo_set:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _clean

        per_combo[combo_set] = (top1, top5)
        attacker_str = ", ".join(sorted(combo_set))
        print(
            f"[InnerLoop]  [{idx}/{len(candidate_pool)}] "
            f"Attackers: [{attacker_str}]  →  Top-1: {top1:.2f}%"
        )

        if top1 < min_top1:
            min_top1 = top1
            min_top5 = top5
            worst_combo = combo_set

    worst_str = ", ".join(sorted(worst_combo)) if worst_combo else "(none)"
    print(
        f"[InnerLoop] Worst: {{{worst_str}}} @ {min_top1:.2f}%  "
        f"(out of {len(candidate_pool)} candidates)"
    )

    del model

    return InnerLoopResult(
        blueprint=blueprint,
        m_attackers=m_attackers,
        min_top1=min_top1,
        min_top5=min_top5,
        worst_combo=worst_combo,
        per_combo=per_combo,
    )
