import sys
sys.path.insert(0, '.')

import os
import csv
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
from microsplit_framework.ga_outer_loop import GAConfig, GAResult, run_ga
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

M_ATTACKERS = [1]

# ── GA hyperparameters ─────────────────────────────────────────────────────────
POPULATION_SIZE = 40
GENERATIONS     = 12

# ── Preprocessing hyperparameters ─────────────────────────────────────────────
N_PREPROCESS_BATCHES = 150
BATCH_SIZE           = 64
N_CLUSTERS           = 100
N_PER_CLASS          = 1


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

    # Preprocessing — once for all (factor, M) combinations
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

    # ── Attack potentials — once for all GA runs ───────────────────────────────
    print("=== Computing attack potentials (once, shared across all GA runs) ===")
    potentials = compute_attack_potentials(
        flat_model, devices, centroids,
        mini_eval_loader, get_layer_fn, set_layer_fn, device,
    )

    os.makedirs("final_results/ga_sweep", exist_ok=True)

    for m in M_ATTACKERS:
        # ── Baseline (no redundancy, budget=0) ────────────────────────────────
        # print(f"\n{'='*60}")
        # print(f"  Baseline — no redundancy, m_attackers={m}")
        # print(f"{'='*60}")
        # baseline_bp     = DeviceBlueprint.from_dict({})
        # baseline_result = evaluate_inner_loop(
        #     base_model   = flat_model,
        #     devices      = devices,
        #     blueprint    = baseline_bp,
        #     m_attackers  = m,
        #     centroids    = centroids,
        #     eval_loader  = mini_eval_loader,
        #     get_layer_fn = get_layer_fn,
        #     set_layer_fn = set_layer_fn,
        #     device       = device,
        #     potentials   = potentials,
        #     verbose      = True,
        # )
        # baseline_avg_top1 = round(
        #     sum(t1 for t1, _ in baseline_result.per_combo.values()) / len(baseline_result.per_combo), 2
        # )
        # worst_str = ", ".join(sorted(baseline_result.worst_combo))
        # print(f"\nBaseline Worst-Case Top-1: {baseline_result.min_top1:.2f}%")
        # print(f"Baseline Worst-Case Top-5: {baseline_result.min_top5:.2f}%")
        # print(f"Baseline Avg-Case Top-1  : {baseline_avg_top1:.2f}%")
        # print(f"Worst Attacker           : [{worst_str}]")
        # summary_worst[0.0][m] = baseline_result.min_top1
        # summary_avg[0.0][m]   = baseline_avg_top1

        # baseline_dir  = f"final_results/ga_sweep/b0.00_m{m}"
        # os.makedirs(baseline_dir, exist_ok=True)
        # baseline_path = f"{baseline_dir}/baseline.csv"
        # with open(baseline_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["attacker", "top1", "top5"])
        #     for combo, (t1, t5) in sorted(baseline_result.per_combo.items(), key=lambda x: x[1][0]):
        #         combo_str = " + ".join(sorted(combo))
        #         writer.writerow([combo_str, f"{t1:.2f}", f"{t5:.2f}"])
        #     writer.writerow(["AVERAGE", f"{baseline_avg_top1:.2f}", ""])
        # print(f"Baseline results saved to {baseline_path}")

        # print(f"\n[Final validation] baseline, m={m} ...")
        # baseline_final_combos = exhaustive_evaluate(
        #     base_model   = flat_model,
        #     devices      = devices,
        #     blueprint    = baseline_bp,
        #     m_attackers  = m,
        #     centroids    = final_centroids,
        #     eval_loader  = final_eval_loader,
        #     get_layer_fn = get_layer_fn,
        #     set_layer_fn = set_layer_fn,
        #     device       = device,
        # )
        # baseline_final_worst    = min(baseline_final_combos, key=lambda c: baseline_final_combos[c][0])
        # baseline_final_min_top1, baseline_final_min_top5 = baseline_final_combos[baseline_final_worst]
        # baseline_final_avg      = round(
        #     sum(t1 for t1, _ in baseline_final_combos.values()) / len(baseline_final_combos), 2
        # )
        # worst_str_final = ", ".join(sorted(baseline_final_worst))
        # print(f"Final Worst-Case Top-1: {baseline_final_min_top1:.2f}%")
        # print(f"Final Worst-Case Top-5: {baseline_final_min_top5:.2f}%")
        # print(f"Final Avg-Case Top-1  : {baseline_final_avg:.2f}%")
        # print(f"Worst Attacker        : [{worst_str_final}]")
        # summary_final_worst[0.0][m] = baseline_final_min_top1
        # summary_final_avg[0.0][m]   = baseline_final_avg

        # final_baseline_path = f"{baseline_dir}/baseline_final.csv"
        # _write_exhaustive_csv(final_baseline_path, baseline_final_combos)
        # print(f"Final baseline results saved to {final_baseline_path}")

        # ── GA sweep over budget factors ───────────────────────────────────────
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
            param_dir = f"final_results/ga_sweep/b{_bstr(factor)}_m{m}"
            os.makedirs(param_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  GA run — budget={_bstr(factor)}x baseline ({global_budget:,.0f} elements), m={m}")
            print(f"{'='*60}")
            print(f"  Results dir: {param_dir}")
            print(f"  Per-device budgets ({_bstr(device_budget_factor)}x primary partition cost):")
            for dev_id, budget in sorted(device_budgets.items()):
                print(f"    {dev_id:<30} {budget:>12,.0f} elements")

            config = GAConfig(
                global_budget   = global_budget,
                device_budgets  = device_budgets,
                population_size = POPULATION_SIZE,
                generations     = GENERATIONS,
                m_attackers     = m,
            )

            result: GAResult = run_ga(
                base_model       = flat_model,
                devices          = devices,
                dataset          = dataset,
                config           = config,
                get_layer_fn     = get_layer_fn,
                set_layer_fn     = set_layer_fn,
                device           = device,
                potentials       = potentials,
                partition_costs  = partition_costs,
                verbose          = False,
                centroids        = centroids,
                mini_eval_loader = mini_eval_loader,
                seed             = 42,
            )

            worst_str = ", ".join(sorted(result.best_worst_combo))
            print(f"\nBest Blueprint  : {result.best_blueprint}")
            print(f"Worst-Case Top-1: {result.best_min_top1:.2f}%")
            print(f"Worst-Case Top-5: {result.best_min_top5:.2f}%")
            print(f"Avg-Case Top-1  : {result.best_bp_avg_top1:.2f}%")
            print(f"Worst Attacker  : [{worst_str}]")

            hist_path = f"{param_dir}/history.csv"
            with open(hist_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=[
                        "generation", "best_top1", "best_bp_avg_top1",
                        "pop_mean_top1", "best_blueprint", "worst_attacker",
                    ]
                )
                writer.writeheader()
                writer.writerows(result.generation_history)
            print(f"History saved to {hist_path}")

            print(f"\n[Final validation] budget={_bstr(factor)}x, m={m} ...")
            final_combos = exhaustive_evaluate(
                base_model   = flat_model,
                devices      = devices,
                blueprint    = result.best_blueprint,
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
                writer.writerow(["ga_worst_top1", f"{result.best_min_top1:.2f}"])
                writer.writerow(["ga_avg_top1", f"{result.best_bp_avg_top1:.2f}"])
                writer.writerow(["final_worst_top1", f"{final_min_top1:.2f}"])
                writer.writerow(["final_avg_top1", f"{final_avg:.2f}"])
            print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
