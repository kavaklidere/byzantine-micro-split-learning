from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from microsplit_framework.topology import ClientSpec, TopologySpec, DeviceSpec
from microsplit_framework.attacks import CleanAttack
from microsplit_framework.aggregation import AggregationStrategy, MedianAggregation
from microsplit_framework.model import MicrosplitModel
from microsplit_framework.preprocess import preprocess_clustering
from microsplit_framework.ga_inner_loop import (
    DeviceBlueprint,
    AttackPotentials,
    client_specs_from_blueprint,
    extract_centroids,
    get_eval_indices,
    make_balanced_mini_loader,
    evaluate_inner_loop,
)


@dataclass
class GAConfig:
    global_budget:  float                 # B — total extra memory allowed (elements)
    device_budgets: dict[str, float]      # v_d per device_id (extras only, not primary)
    population_size: int
    generations:    int
    m_attackers:    int
    add_weight:     float = 0.4
    remove_weight:  float = 0.2
    swap_weight:    float = 0.4
    elitism_ratio:  float = 0.2
    n_preprocess_batches: int = 150
    n_clusters:     int = 100
    batch_size:     int = 64
    n_per_class:    int = 1
    T_provisional:  int = 15
    aggregation: AggregationStrategy = field(default_factory=MedianAggregation)


@dataclass
class GAResult:
    best_blueprint:   DeviceBlueprint
    best_min_top1:    float
    best_min_top5:    float
    best_bp_avg_top1: float
    best_worst_combo: frozenset
    generation_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _extras_cost(extras: dict, partition_costs: dict) -> float:
    return sum(partition_costs[s] for (s, _) in extras)


def _device_cost(extras: dict, dev_id: str, partition_costs: dict) -> float:
    return sum(partition_costs[s] for (s, d) in extras if d == dev_id)


def _addable_pairs(
    extras: dict,
    all_pairs: list,
    config: GAConfig,
    partition_costs: dict,
) -> list:
    total = _extras_cost(extras, partition_costs)
    max_extras_per_partition = 2 * config.m_attackers

    extras_per_partition: dict[str, int] = {}
    for (s, _) in extras:
        extras_per_partition[s] = extras_per_partition.get(s, 0) + 1

    out = []
    for (s, d) in all_pairs:
        if (s, d) in extras:
            continue
        if extras_per_partition.get(s, 0) >= max_extras_per_partition:
            continue
        c = partition_costs[s]
        if total + c > config.global_budget:
            continue
        if _device_cost(extras, d, partition_costs) + c > config.device_budgets.get(d, float("inf")):
            continue
        out.append((s, d))
    return out


def _sample_cheap(pairs: list, partition_costs: dict) -> tuple:
    weights = [1.0 / partition_costs[s] for (s, _) in pairs]
    return random.choices(pairs, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _generate_individual(all_pairs: list, config: GAConfig, partition_costs: dict) -> dict:
    extras: dict = {}
    while True:
        addable = _addable_pairs(extras, all_pairs, config, partition_costs)
        if not addable:
            break
        s, d = _sample_cheap(addable, partition_costs)
        extras[(s, d)] = 1
    return extras


def _mutate(
    bp: DeviceBlueprint,
    all_pairs: list,
    config: GAConfig,
    partition_costs: dict,
) -> DeviceBlueprint:
    extras = dict(bp.all_assignments())
    move = random.choices(
        ["add", "remove", "swap"],
        weights=[config.add_weight, config.remove_weight, config.swap_weight],
    )[0]

    if move == "add":
        a = _addable_pairs(extras, all_pairs, config, partition_costs)
        if a:
            s, d = _sample_cheap(a, partition_costs)
            extras[(s, d)] = 1

    elif move == "remove":
        if extras:
            del extras[random.choice(list(extras))]

    elif move == "swap":
        if extras:
            r = random.choice(list(extras))
            del extras[r]
            a = _addable_pairs(extras, all_pairs, config, partition_costs)
            if a:
                s, d = _sample_cheap(a, partition_costs)
                extras[(s, d)] = 1

    return DeviceBlueprint.from_dict(extras)


def _crossover(
    pa: DeviceBlueprint,
    pb: DeviceBlueprint,
    all_pairs: list,
    config: GAConfig,
    partition_costs: dict,
    value_per_cost: dict,
) -> DeviceBlueprint:
    # Stage 1 — inherit: uniform crossover over union of both parents
    extras: dict = {}
    for key in set(pa.all_assignments()) | set(pb.all_assignments()):
        if random.random() < 0.5:
            extras[key] = 1

    # Stage 1.5 — remove over-protected partitions (> 2m extras per partition)
    max_extras_per_partition = 2 * config.m_attackers
    partition_pairs: dict[str, list] = {}
    for key in extras:
        s, _ = key
        partition_pairs.setdefault(s, []).append(key)
    for s, pairs in partition_pairs.items():
        while len(pairs) > max_extras_per_partition:
            # Prefer removing from a device that is already over its budget — this
            # simultaneously fixes over-protection and reduces Stage 2a repair work.
            # Falls back to random when no over-budget device holds an excess replica.
            over_budget = [
                p for p in pairs
                if _device_cost(extras, p[1], partition_costs) > config.device_budgets.get(p[1], float("inf"))
            ]
            to_remove = random.choice(over_budget) if over_budget else random.choice(pairs)
            del extras[to_remove]
            pairs.remove(to_remove)

    # Stage 2a — repair per-device violations (local, fix before global)
    for dev_id, v_d in config.device_budgets.items():
        while _device_cost(extras, dev_id, partition_costs) > v_d:
            on_dev = [(s, d) for (s, d) in extras if d == dev_id]
            worst = min(on_dev, key=lambda k: value_per_cost.get(k[0], 0.0))
            del extras[worst]

    # Stage 2b — repair global budget
    while _extras_cost(extras, partition_costs) > config.global_budget:
        worst = min(extras, key=lambda k: value_per_cost.get(k[0], 0.0))
        del extras[worst]

    # Stage 3 — refill any slack opened by repair
    while True:
        a = _addable_pairs(extras, all_pairs, config, partition_costs)
        if not a:
            break
        s, d = _sample_cheap(a, partition_costs)
        extras[(s, d)] = 1

    return DeviceBlueprint.from_dict(extras)


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(
    base_model: nn.Module,
    devices: list[DeviceSpec],
    dataset,
    config: GAConfig,
    get_layer_fn: Callable,
    set_layer_fn: Callable,
    device: torch.device,
    potentials: AttackPotentials,
    partition_costs: dict[str, float],
    verbose: bool = False,
    centroids: dict | None = None,
    mini_eval_loader: DataLoader | None = None,
    seed: int = 42,
) -> GAResult:
    """
    Budget-aware minimax GA: maximises worst-case Top-1 accuracy under m_attackers
    simultaneously compromised devices.

    Set semantics: each (partition, device) pair appears at most once per blueprint.
    Primary (σ, d) pairs (the baseline assignment on each device) are excluded from
    the search space entirely.

    Feasibility constraints:
      - sum of f(σ) for all extras ≤ global_budget (B)
      - sum of f(σ) for extras on device d ≤ device_budgets[d] (v_d)

    partition_costs: dict[client_id → total_elements] from compute_memory_costs().
    potentials: pre-computed via compute_attack_potentials().
    If centroids and mini_eval_loader are both provided, preprocessing is skipped.
    """
    if potentials is None:
        raise ValueError(
            "potentials must be provided. "
            "Call compute_attack_potentials() before run_ga and pass the result."
        )
    random.seed(seed)

    device_ids    = [d.device_id for d in devices]
    partition_ids = [cs.client_id for d in devices for cs in d.client_specs]

    # Primary (σ, d) pairs are excluded — σ is already on d as the baseline replica
    primary_pairs = {
        (cs.client_id, dev.device_id)
        for dev in devices
        for cs in dev.client_specs
    }
    all_pairs = [
        (s, dv)
        for dv in device_ids
        for s in partition_ids
        if (s, dv) not in primary_pairs
    ]

    # value_per_cost[σ] = p_σ / f(σ) — used for eviction priority in crossover repair
    value_per_cost: dict[str, float] = {
        s: potentials.potentials.get(s, 0.0) / max(partition_costs.get(s, 1.0), 1.0)
        for s in partition_ids
    }

    # --- One-time preprocessing (skipped if centroids + mini_eval_loader supplied) ---
    if centroids is None or mini_eval_loader is None:
        eval_idx_set = set(get_eval_indices(dataset, config.n_per_class, seed=42))
        pre_pool = [i for i in range(len(dataset)) if i not in eval_idx_set]
        import random as _random
        _random.Random(42).shuffle(pre_pool)
        n_pre = config.n_preprocess_batches * config.batch_size
        _g = torch.Generator()
        _g.manual_seed(seed)
        pre_loader = DataLoader(
            Subset(dataset, pre_pool[:n_pre]),
            batch_size=config.batch_size, shuffle=True, num_workers=2, generator=_g,
        )
        mini_eval_loader = make_balanced_mini_loader(
            dataset, config.n_per_class, config.batch_size, seed=42,
        )
        print(f"[GA] Mini-eval: {len(mini_eval_loader.dataset)} images")

        baseline_specs = client_specs_from_blueprint(devices, DeviceBlueprint.from_dict({}), device_ids)
        preprocess_topology = TopologySpec(clients=[
            ClientSpec(
                client_id=cs.client_id, layer_range=cs.layer_range,
                height_range=cs.height_range, replicas=[CleanAttack()],
                aggregation=config.aggregation,
                replica_device_ids=cs.replica_device_ids,
            )
            for cs in baseline_specs
        ])
        preprocess_model = MicrosplitModel(
            base_model, preprocess_topology, get_layer_fn, set_layer_fn,
        ).to(device)
        print("[GA] Running preprocessing (KMeans)...")
        preprocess_clustering(
            preprocess_model, pre_loader,
            n_batches=config.n_preprocess_batches, n_clusters=config.n_clusters, device=device,
        )
        centroids = extract_centroids(preprocess_model)
        del preprocess_model
        print(f"[GA] Centroids ready for: {list(centroids.keys())}\n")
    else:
        print(f"[GA] Using pre-computed centroids for: {list(centroids.keys())}\n")

    # Fitness cache — DeviceBlueprint is a frozen dataclass (hashable)
    fitness_cache: dict[DeviceBlueprint, tuple] = {}

    def get_fitness(bp: DeviceBlueprint) -> tuple:
        if bp not in fitness_cache:
            result = evaluate_inner_loop(
                base_model=base_model,
                devices=devices,
                blueprint=bp,
                m_attackers=config.m_attackers,
                centroids=centroids,
                eval_loader=mini_eval_loader,
                get_layer_fn=get_layer_fn,
                set_layer_fn=set_layer_fn,
                device=device,
                potentials=potentials,
                T_provisional=config.T_provisional,
                aggregation=config.aggregation,
                verbose=verbose,
            )
            avg_top1 = round(
                sum(t1 for t1, _ in result.per_combo.values()) / len(result.per_combo), 2
            )
            fitness_cache[bp] = (result.min_top1, result.min_top5, avg_top1, result.worst_combo)
        return fitness_cache[bp]

    # --- Initial population (all grown from scratch) ---
    population: list[DeviceBlueprint] = [
        DeviceBlueprint.from_dict(_generate_individual(all_pairs, config, partition_costs))
        for _ in range(config.population_size)
    ]
    n_parents = max(2, int(config.population_size * config.elitism_ratio))

    best_blueprint: DeviceBlueprint | None = None
    best_min_top1   = -float("inf")
    best_min_top5   = -float("inf")
    best_bp_avg_top1 = -float("inf")
    best_worst_combo: frozenset = frozenset()
    history: list[dict] = []

    for gen in range(config.generations):
        n_cached = sum(1 for bp in population if bp in fitness_cache)
        n_fresh  = len(population) - n_cached
        print(
            f"\n[GA] ── Gen {gen+1}/{config.generations} "
            f"({n_fresh} to evaluate, {n_cached} cached) ──────────────"
        )

        scored = [(bp, *get_fitness(bp)) for bp in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        gen_best_bp, gen_best_top1, gen_best_top5, gen_best_avg, gen_best_worst = scored[0]
        gen_pop_mean = sum(s[1] for s in scored) / len(scored)

        if gen_best_top1 > best_min_top1:
            best_min_top1    = gen_best_top1
            best_min_top5    = gen_best_top5
            best_bp_avg_top1 = gen_best_avg
            best_blueprint   = gen_best_bp
            best_worst_combo = gen_best_worst

        worst_str = " + ".join(sorted(gen_best_worst))
        history.append({
            "generation":       gen + 1,
            "best_top1":        gen_best_top1,
            "pop_mean_top1":    round(gen_pop_mean, 2),
            "best_bp_avg_top1": gen_best_avg,
            "best_blueprint":   str(gen_best_bp),
            "worst_attacker":   worst_str,
        })
        print(
            f"[GA] Gen {gen+1:>2}/{config.generations} | "
            f"Best Top-1: {gen_best_top1:.2f}% | "
            f"Avg (top-T): {gen_best_avg:.2f}% | "
            f"Mean: {gen_pop_mean:.2f}% | "
            f"{gen_best_bp}"
        )

        if gen == config.generations - 1:
            break

        survivors = [bp for bp, _, _, _, _ in scored[:n_parents]]
        new_population: list[DeviceBlueprint] = list(survivors)
        while len(new_population) < config.population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = _crossover(
                parent1, parent2, all_pairs, config, partition_costs, value_per_cost,
            )
            child = _mutate(child, all_pairs, config, partition_costs)
            new_population.append(child)
        population = new_population

    return GAResult(
        best_blueprint=best_blueprint,
        best_min_top1=best_min_top1,
        best_min_top5=best_min_top5,
        best_bp_avg_top1=best_bp_avg_top1,
        best_worst_combo=best_worst_combo,
        generation_history=history,
    )
