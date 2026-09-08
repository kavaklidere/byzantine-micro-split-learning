import sys
sys.path.insert(0, '.')

import os
import csv
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models

from microsplit_framework import (
    make_flat_sequential,
    DeviceBlueprint,
    compute_attack_potentials,
    evaluate_inner_loop,
    compute_memory_costs,
    client_specs_from_blueprint,
)
from evaluate_blueprint import exhaustive_evaluate, write_csv as _write_exhaustive_csv
from sweep_utils import build_devices, run_preprocessing


# ── Sweep parameters ──────────────────────────────────────────────────────────
# Global budget as a multiple of baseline cost (sum of all primary partition costs).
# E.g. 1.5 means extra replicas can cost at most 1.5× the baseline memory total.
GLOBAL_BUDGET_FACTORS = [0.5, 1.0, 1.5, 2.0]

# Per-device budget as a multiple of that device's own primary partition cost.
# Tied to the global budget factor each iteration (see loop below), rather than
# a single fixed value for the whole sweep.
# E.g. 1.0 means each device can host extras costing at most 1× its primary partition.

M_ATTACKERS   = [1]
T_PROVISIONAL = 15


# ── Budget helpers ─────────────────────────────────────────────────────────────

def _extras_cost(extras: dict, partition_costs: dict) -> float:
    return sum(partition_costs[s] for (s, _) in extras)


def _device_cost(extras: dict, dev_id: str, partition_costs: dict) -> float:
    return sum(partition_costs[s] for (s, d) in extras if d == dev_id)


def _addable_pairs(
    extras: dict,
    all_pairs: list,
    global_budget: float,
    device_budgets: dict,
    partition_costs: dict,
) -> list:
    total = _extras_cost(extras, partition_costs)
    out = []
    for (s, d) in all_pairs:
        if (s, d) in extras:
            continue
        c = partition_costs[s]
        if total + c > global_budget:
            continue
        if _device_cost(extras, d, partition_costs) + c > device_budgets.get(d, float("inf")):
            continue
        out.append((s, d))
    return out


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class GreedyStepResult:
    step:            int
    chosen_partition: str
    chosen_device:   str
    gain:            float    # fitness(π ∪ {(σ*,d*)}) − fitness(π)
    score:           float    # gain / partition_costs[σ*]
    cost:            float    # partition_costs[σ*]
    total_cost_used: float    # _extras_cost of updated π
    min_top1:        float
    min_top5:        float
    avg_top1:        float
    worst_combo:     frozenset
    blueprint:       DeviceBlueprint


@dataclass
class GreedyResult:
    final_blueprint:   DeviceBlueprint
    final_min_top1:    float
    final_min_top5:    float
    final_avg_top1:    float
    final_worst_combo: frozenset
    step_history:      list = field(default_factory=list)


# ── Core greedy algorithm ──────────────────────────────────────────────────────

def run_greedy(
    base_model,
    devices,
    all_pairs: list,
    global_budget: float,
    device_budgets: dict,
    m_attackers: int,
    partition_costs: dict,
    centroids,
    mini_eval_loader,
    get_layer_fn,
    set_layer_fn,
    device,
    potentials,
    T_provisional: int = 15,
) -> GreedyResult:
    extras: dict = {}
    history: list[GreedyStepResult] = []

    # Evaluate π = ∅ once to get the baseline fitness for gain computation
    init_result = evaluate_inner_loop(
        base_model   = base_model,
        devices      = devices,
        blueprint    = DeviceBlueprint.from_dict({}),
        m_attackers  = m_attackers,
        centroids    = centroids,
        eval_loader  = mini_eval_loader,
        get_layer_fn = get_layer_fn,
        set_layer_fn = set_layer_fn,
        device       = device,
        potentials   = potentials,
        T_provisional= T_provisional,
        verbose      = False,
    )
    cur_top1 = init_result.min_top1

    step = 0
    while True:
        addable = _addable_pairs(extras, all_pairs, global_budget, device_budgets, partition_costs)
        if not addable:
            break
        step += 1
        print(f"\n[Greedy] Step {step} — evaluating {len(addable)} candidates ...")

        best_score  = -float("inf")
        best_pair   = None
        best_result = None

        for (s, d) in addable:
            trial = dict(extras)
            trial[(s, d)] = 1
            r = evaluate_inner_loop(
                base_model   = base_model,
                devices      = devices,
                blueprint    = DeviceBlueprint.from_dict(trial),
                m_attackers  = m_attackers,
                centroids    = centroids,
                eval_loader  = mini_eval_loader,
                get_layer_fn = get_layer_fn,
                set_layer_fn = set_layer_fn,
                device       = device,
                potentials   = potentials,
                T_provisional= T_provisional,
                verbose      = False,
            )
            gain  = r.min_top1 - cur_top1
            score = gain / partition_costs[s]
            if score > best_score:
                best_score  = score
                best_pair   = (s, d)
                best_result = r

        s_star, d_star = best_pair
        extras[(s_star, d_star)] = 1
        cur_top1 = best_result.min_top1

        gain_val = best_score * partition_costs[s_star]
        avg_top1 = round(
            sum(t1 for t1, _ in best_result.per_combo.values()) / len(best_result.per_combo), 2
        )
        total_cost = _extras_cost(extras, partition_costs)

        rec = GreedyStepResult(
            step             = step,
            chosen_partition = s_star,
            chosen_device    = d_star,
            gain             = gain_val,
            score            = best_score,
            cost             = partition_costs[s_star],
            total_cost_used  = total_cost,
            min_top1         = best_result.min_top1,
            min_top5         = best_result.min_top5,
            avg_top1         = avg_top1,
            worst_combo      = best_result.worst_combo,
            blueprint        = DeviceBlueprint.from_dict(extras),
        )
        history.append(rec)
        print(
            f"[Greedy] Step {step} → added ({s_star}, {d_star})  "
            f"gain: {gain_val:+.4f}  score: {best_score:.6f}  "
            f"cost used: {total_cost:,.0f}  worst-case Top-1: {best_result.min_top1:.2f}%"
        )

    final_bp = DeviceBlueprint.from_dict(extras)
    if history:
        last = history[-1]
        return GreedyResult(
            final_blueprint   = final_bp,
            final_min_top1    = last.min_top1,
            final_min_top5    = last.min_top5,
            final_avg_top1    = last.avg_top1,
            final_worst_combo = last.worst_combo,
            step_history      = history,
        )
    # Budget was 0 — no steps taken; return empty blueprint with init fitness
    init_avg = round(
        sum(t1 for t1, _ in init_result.per_combo.values()) / len(init_result.per_combo), 2
    )
    return GreedyResult(
        final_blueprint   = final_bp,
        final_min_top1    = init_result.min_top1,
        final_min_top5    = init_result.min_top5,
        final_avg_top1    = init_avg,
        final_worst_combo = init_result.worst_combo,
        step_history      = history,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def _bstr(factor: float) -> str:
    return f"{factor:.2f}"


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    weights  = models.VGG16_Weights.DEFAULT
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

    devices    = build_devices()
    device_ids = [d.device_id for d in devices]

    centroids, mini_eval_loader, final_centroids, final_eval_loader = run_preprocessing(
        flat_model, devices, device_ids, dataset, get_layer_fn, set_layer_fn, device,
    )

    # ── Memory costs — computed once from the K=0 topology ────────────────────
    baseline_specs  = client_specs_from_blueprint(devices, DeviceBlueprint.from_dict({}), device_ids)
    mem_costs       = compute_memory_costs(flat_model, baseline_specs)
    partition_costs = {cid: float(v["total_elements"]) for cid, v in mem_costs.items()}
    baseline_cost   = sum(partition_costs.values())

    print("=== Partition memory costs ===")
    for cid, cost in sorted(partition_costs.items(), key=lambda x: -x[1]):
        print(f"  {cid:<30} {cost:>12,.0f} elements")
    print(f"  {'BASELINE TOTAL':<30} {baseline_cost:>12,.0f} elements\n")
    for factor in GLOBAL_BUDGET_FACTORS:
        print(f"  global_budget ({_bstr(factor)}x) = {factor * baseline_cost:>12,.0f} elements")
    print()

    # ── Attack potentials — once for all greedy runs ───────────────────────────
    print("=== Computing attack potentials (once, shared across all greedy runs) ===")
    potentials = compute_attack_potentials(
        flat_model, devices, centroids,
        mini_eval_loader, get_layer_fn, set_layer_fn, device,
    )

    # ── Candidate pairs — exclude primary (σ, d) assignments ─────────────────
    partition_ids = [cs.client_id for d in devices for cs in d.client_specs]
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
    print(f"Candidate (σ, d) pairs: {len(all_pairs)} (= {len(partition_ids)} × {len(device_ids)} − {len(primary_pairs)} primary)\n")

    os.makedirs("final_results/greedy_sweep", exist_ok=True)

    for m in M_ATTACKERS:
        for factor in GLOBAL_BUDGET_FACTORS:
            global_budget = factor * baseline_cost

            # Per-device budget factor tracks the global budget factor for this run.
            device_budget_factor = factor
            device_budgets: dict[str, float] = {
                dev.device_id: device_budget_factor * sum(
                    partition_costs[cs.client_id] for cs in dev.client_specs
                )
                for dev in devices
            }

            # One folder per (budget factor, m) combo — keeps every sweep's files
            # isolated so re-running with a different parameter set never overwrites
            # another parameter's results.
            param_dir = f"final_results/greedy_sweep/b{_bstr(factor)}_m{m}"
            os.makedirs(param_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  Greedy run — budget={_bstr(factor)}x baseline ({global_budget:,.0f} elements), m={m}")
            print(f"{'='*60}")
            print(f"  Results dir: {param_dir}")
            print(f"  Per-device budgets ({_bstr(device_budget_factor)}x primary partition cost):")
            for dev_id, budget in sorted(device_budgets.items()):
                print(f"    {dev_id:<30} {budget:>12,.0f} elements")

            greedy_result = run_greedy(
                base_model      = flat_model,
                devices         = devices,
                all_pairs       = all_pairs,
                global_budget   = global_budget,
                device_budgets  = device_budgets,
                m_attackers     = m,
                partition_costs = partition_costs,
                centroids       = centroids,
                mini_eval_loader= mini_eval_loader,
                get_layer_fn    = get_layer_fn,
                set_layer_fn    = set_layer_fn,
                device          = device,
                potentials      = potentials,
                T_provisional   = T_PROVISIONAL,
            )

            worst_str = ", ".join(sorted(greedy_result.final_worst_combo))
            print(f"\nFinal Blueprint   : {greedy_result.final_blueprint}")
            print(f"Steps taken       : {len(greedy_result.step_history)}")
            print(f"Worst-Case Top-1  : {greedy_result.final_min_top1:.2f}%")
            print(f"Worst-Case Top-5  : {greedy_result.final_min_top5:.2f}%")
            print(f"Avg-Case Top-1    : {greedy_result.final_avg_top1:.2f}%")
            print(f"Worst Attacker    : [{worst_str}]")

            hist_path = f"{param_dir}/history.csv"
            with open(hist_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "chosen_partition", "chosen_device",
                    "gain", "score", "cost", "total_cost_used",
                    "min_top1", "min_top5", "avg_top1", "worst_attacker", "blueprint",
                ])
                for rec in greedy_result.step_history:
                    writer.writerow([
                        rec.step,
                        rec.chosen_partition,
                        rec.chosen_device,
                        f"{rec.gain:.6f}",
                        f"{rec.score:.8f}",
                        f"{rec.cost:.0f}",
                        f"{rec.total_cost_used:.0f}",
                        f"{rec.min_top1:.2f}",
                        f"{rec.min_top5:.2f}",
                        f"{rec.avg_top1:.2f}",
                        " + ".join(sorted(rec.worst_combo)),
                        str(rec.blueprint),
                    ])
            print(f"History saved to {hist_path}")

            print(f"\n[Final validation] budget={_bstr(factor)}x, m={m} ...")
            final_combos = exhaustive_evaluate(
                base_model   = flat_model,
                devices      = devices,
                blueprint    = greedy_result.final_blueprint,
                m_attackers  = m,
                centroids    = final_centroids,
                eval_loader  = final_eval_loader,
                get_layer_fn = get_layer_fn,
                set_layer_fn = set_layer_fn,
                device       = device,
            )
            final_worst_combo   = min(final_combos, key=lambda c: final_combos[c][0])
            final_min_top1, final_min_top5 = final_combos[final_worst_combo]
            final_avg           = round(
                sum(t1 for t1, _ in final_combos.values()) / len(final_combos), 2
            )
            worst_str_final = ", ".join(sorted(final_worst_combo))
            print(f"Final Worst-Case Top-1: {final_min_top1:.2f}%")
            print(f"Final Worst-Case Top-5: {final_min_top5:.2f}%")
            print(f"Final Avg-Case Top-1  : {final_avg:.2f}%")
            print(f"Worst Attacker        : [{worst_str_final}]")

            final_path = f"{param_dir}/final.csv"
            _write_exhaustive_csv(final_path, final_combos)
            print(f"Final results saved to {final_path}")

            summary_path = f"{param_dir}/summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["budget_factor", _bstr(factor)])
                writer.writerow(["device_budget_factor", _bstr(device_budget_factor)])
                writer.writerow(["m_attackers", m])
                writer.writerow(["greedy_worst_top1", f"{greedy_result.final_min_top1:.2f}"])
                writer.writerow(["greedy_avg_top1", f"{greedy_result.final_avg_top1:.2f}"])
                writer.writerow(["final_worst_top1", f"{final_min_top1:.2f}"])
                writer.writerow(["final_avg_top1", f"{final_avg:.2f}"])
            print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
