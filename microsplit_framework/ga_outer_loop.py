from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from microsplit_framework.topology import ClientSpec, TopologySpec
from microsplit_framework.attacks import CleanAttack
from microsplit_framework.aggregation import AggregationStrategy, MedianAggregation
from microsplit_framework.model import MicrosplitModel
from microsplit_framework.preprocess import preprocess_clustering
from microsplit_framework.ga_inner_loop import (
    RedundancyBlueprint,
    extract_centroids,
    get_eval_indices,
    make_balanced_mini_loader,
    evaluate_inner_loop,
)


@dataclass
class GAConfig:
    budget: int               # K: redundancy extras; total replicas = K + n_clients
    population_size: int
    generations: int
    m_attackers: int
    mutation_rate: float = 0.2
    elitism_ratio: float = 0.2
    aggregation: AggregationStrategy = field(default_factory=MedianAggregation)
    n_preprocess_batches: int = 150
    n_clusters: int = 100
    batch_size: int = 64
    n_per_class: int = 1
    n_seed_from_prev: int = 0  # how many initial population slots come from seed_blueprints


@dataclass
class GAResult:
    best_blueprint: RedundancyBlueprint
    best_min_top1: float
    best_min_top5: float
    best_bp_avg_top1: float
    best_worst_combo: frozenset
    generation_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Genetic operators — all work in "extras" space and preserve sum(extras) == K
# ---------------------------------------------------------------------------

def _generate_blueprint(client_ids: list[str], budget: int) -> RedundancyBlueprint:
    extras: dict[str, int] = {}
    for _ in range(budget):
        cid = random.choice(client_ids)
        extras[cid] = extras.get(cid, 0) + 1
    return RedundancyBlueprint.from_dict(extras)


def _crossover(
    pa: RedundancyBlueprint,
    pb: RedundancyBlueprint,
    client_ids: list[str],
    budget: int,
) -> RedundancyBlueprint:
    extras = {
        cid: random.choice([pa.n_replicas(cid) - 1, pb.n_replicas(cid) - 1])
        for cid in client_ids
    }
    total = sum(extras.values())
    while total > budget:
        donors = [cid for cid in client_ids if extras[cid] > 0]
        extras[random.choice(donors)] -= 1
        total -= 1
    while total < budget:
        extras[random.choice(client_ids)] += 1
        total += 1
    return RedundancyBlueprint.from_dict({cid: e for cid, e in extras.items() if e > 0})


def _mutate(
    blueprint: RedundancyBlueprint,
    client_ids: list[str],
    mutation_rate: float,
) -> RedundancyBlueprint:
    if random.random() >= mutation_rate:
        return blueprint
    extras = {cid: blueprint.n_replicas(cid) - 1 for cid in client_ids}
    donors = [cid for cid in client_ids if extras[cid] > 0]
    if not donors:
        return blueprint
    donor = random.choice(donors)
    extras[donor] -= 1
    recipients = [cid for cid in client_ids if cid != donor]
    extras[random.choice(recipients)] += 1
    return RedundancyBlueprint.from_dict({cid: e for cid, e in extras.items() if e > 0})


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(
    base_model: nn.Module,
    base_client_specs: list[ClientSpec],
    dataset,
    config: GAConfig,
    get_layer_fn: Callable,
    set_layer_fn: Callable,
    device: torch.device,
    verbose: bool = False,
    centroids: dict | None = None,
    mini_eval_loader: DataLoader | None = None,
    seed: int = 42,
    seed_blueprints: list[RedundancyBlueprint] | None = None,
) -> GAResult:
    """
    Minimax GA: maximise worst-case Top-1 accuracy under m_attackers simultaneous
    malicious replicas, over all RedundancyBlueprints with exactly `budget` extra
    replicas distributed across clients.

    If centroids and mini_eval_loader are both provided, preprocessing is skipped —
    useful when running multiple GA sweeps sharing one preprocessing pass.
    """
    random.seed(seed)
    client_ids = [spec.client_id for spec in base_client_specs]

    # --- One-time preprocessing (skipped if centroids + mini_eval_loader supplied) ---
    if centroids is None or mini_eval_loader is None:
        eval_idx_set = set(get_eval_indices(dataset, config.n_per_class, seed=42))
        pre_pool = [i for i in range(len(dataset)) if i not in eval_idx_set]
        n_pre = config.n_preprocess_batches * config.batch_size
        import random as _random
        _random.Random(42).shuffle(pre_pool)
        _g = torch.Generator()
        _g.manual_seed(seed)
        pre_loader = DataLoader(
            Subset(dataset, pre_pool[:n_pre]),
            batch_size=config.batch_size, shuffle=True, num_workers=2, generator=_g,
        )
        mini_eval_loader = make_balanced_mini_loader(dataset, config.n_per_class, config.batch_size, seed=42)
        print(f"[GA] Mini-eval: {len(mini_eval_loader.dataset)} images ({len(eval_idx_set)} reserved, no overlap with preprocessing)")

        preprocess_topology = TopologySpec(clients=[
            ClientSpec(
                client_id=spec.client_id, layer_range=spec.layer_range,
                height_range=spec.height_range, replicas=[CleanAttack()],
                aggregation=config.aggregation,
            )
            for spec in base_client_specs
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

    # Fitness cache — RedundancyBlueprint is a frozen dataclass (hashable)
    fitness_cache: dict[RedundancyBlueprint, tuple[float, float, float, frozenset]] = {}

    def get_fitness(bp: RedundancyBlueprint) -> tuple[float, float, float, frozenset]:
        if bp not in fitness_cache:
            result = evaluate_inner_loop(
                base_model=base_model,
                base_client_specs=base_client_specs,
                blueprint=bp,
                m_attackers=config.m_attackers,
                centroids=centroids,
                mini_eval_loader=mini_eval_loader,
                get_layer_fn=get_layer_fn,
                set_layer_fn=set_layer_fn,
                device=device,
                aggregation=config.aggregation,
                verbose=verbose,
            )
            best_bp_avg_top1 = round(sum(t1 for t1, _ in result.per_combo.values()) / len(result.per_combo), 2)
            fitness_cache[bp] = (result.min_top1, result.min_top5, best_bp_avg_top1, result.worst_combo)
        return fitness_cache[bp]

    # --- Initial population ---
    population: list[RedundancyBlueprint] = []
    if seed_blueprints and config.n_seed_from_prev > 0:
        n_from_prev = min(config.n_seed_from_prev, len(seed_blueprints), config.population_size)
        population.extend(random.sample(seed_blueprints, n_from_prev))
    population += [_generate_blueprint(client_ids, config.budget) for _ in range(config.population_size - len(population))]
    n_parents = max(2, int(config.population_size * config.elitism_ratio))

    best_blueprint: RedundancyBlueprint | None = None
    best_min_top1 = -float("inf")
    best_min_top5 = -float("inf")
    best_best_bp_avg_top1 = -float("inf")
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

        gen_best_bp, gen_best_top1, gen_best_top5, gen_best_best_bp_avg_top1, gen_best_worst_combo = scored[0]
        gen_pop_mean_top1 = sum(s[1] for s in scored) / len(scored)

        if gen_best_top1 > best_min_top1:
            best_min_top1    = gen_best_top1
            best_min_top5    = gen_best_top5
            best_best_bp_avg_top1    = gen_best_best_bp_avg_top1
            best_blueprint   = gen_best_bp
            best_worst_combo = gen_best_worst_combo

        worst_attacker_str = " + ".join(f"{cid}[{i}]" for cid, i in sorted(gen_best_worst_combo))
        history.append({
            "generation":     gen + 1,
            "best_top1":      gen_best_top1,
            "pop_mean_top1":      round(gen_pop_mean_top1, 2),
            "best_bp_avg_top1":       gen_best_best_bp_avg_top1,
            "best_blueprint": str(gen_best_bp),
            "worst_attacker": worst_attacker_str,
        })
        print(
            f"[GA] Gen {gen+1:>2}/{config.generations} | "
            f"Best Top-1: {gen_best_top1:.2f}% | "
            f"Avg: {gen_best_best_bp_avg_top1:.2f}% | "
            f"Mean: {gen_pop_mean_top1:.2f}% | "
            f"{gen_best_bp}"
        )

        if gen == config.generations - 1:
            break

        survivors = [bp for bp, _, _, _, _ in scored[:n_parents]]
        new_population: list[RedundancyBlueprint] = list(survivors)
        while len(new_population) < config.population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = _crossover(parent1, parent2, client_ids, config.budget)
            child = _mutate(child, client_ids, config.mutation_rate)
            new_population.append(child)
        population = new_population

    return GAResult(
        best_blueprint=best_blueprint,
        best_min_top1=best_min_top1,
        best_min_top5=best_min_top5,
        best_bp_avg_top1=best_best_bp_avg_top1,
        best_worst_combo=best_worst_combo,
        generation_history=history,
    )
