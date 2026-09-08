import random as _random

import torch
from torch.utils.data import DataLoader, Subset

from microsplit_framework import (
    ClientSpec, CleanAttack, MicrosplitModel, TopologySpec,
    DeviceBlueprint, DeviceSpec, client_specs_from_blueprint,
    preprocess_clustering, get_eval_indices, make_balanced_mini_loader, extract_centroids,
)

# ── Preprocessing hyperparameters (shared defaults) ───────────────────────────
N_PREPROCESS_BATCHES = 150
BATCH_SIZE           = 64
N_CLUSTERS           = 100
N_PER_CLASS          = 1


def build_devices() -> list[DeviceSpec]:
    """
    10-device topology: conv1+2+3 (layers 0-16) in 3 horizontal stripes,
    conv4+5 (layers 17-31) in 4 horizontal stripes,
    dense1+2 (layers 32-39) in 3 horizontal stripes.
    """
    t = 1 / 3
    return [
        # conv1+2+3 — layers 0-16, 3 stripes
        DeviceSpec("device_conv123_top", [ClientSpec("conv123_top", layer_range=(0,  16), height_range=(0.0, t),   replicas=[CleanAttack()])]),
        DeviceSpec("device_conv123_mid", [ClientSpec("conv123_mid", layer_range=(0,  16), height_range=(t,   2*t), replicas=[CleanAttack()])]),
        DeviceSpec("device_conv123_bot", [ClientSpec("conv123_bot", layer_range=(0,  16), height_range=(2*t, 1.0), replicas=[CleanAttack()])]),
        # conv4+5 — layers 17-31, 4 stripes
        DeviceSpec("device_conv45_s0",   [ClientSpec("conv45_s0",   layer_range=(17, 31), height_range=(0.0,  0.25), replicas=[CleanAttack()])]),
        DeviceSpec("device_conv45_s1",   [ClientSpec("conv45_s1",   layer_range=(17, 31), height_range=(0.25, 0.5),  replicas=[CleanAttack()])]),
        DeviceSpec("device_conv45_s2",   [ClientSpec("conv45_s2",   layer_range=(17, 31), height_range=(0.5,  0.75), replicas=[CleanAttack()])]),
        DeviceSpec("device_conv45_s3",   [ClientSpec("conv45_s3",   layer_range=(17, 31), height_range=(0.75, 1.0),  replicas=[CleanAttack()])]),
        # dense1+2 — layers 32-39, 3 stripes
        DeviceSpec("device_dense12_top", [ClientSpec("dense12_top", layer_range=(32, 39), height_range=(0.0, t),   replicas=[CleanAttack()])]),
        DeviceSpec("device_dense12_mid", [ClientSpec("dense12_mid", layer_range=(32, 39), height_range=(t,   2*t), replicas=[CleanAttack()])]),
        DeviceSpec("device_dense12_bot", [ClientSpec("dense12_bot", layer_range=(32, 39), height_range=(2*t, 1.0), replicas=[CleanAttack()])]),
    ]


def run_preprocessing(flat_model, devices, device_ids, dataset, get_layer_fn, set_layer_fn, device):
    """
    Build four mutually disjoint data splits:
      1) GA warm-up        — KMeans clustering for the GA's cluster-jump attackers
      2) GA mini-eval       — small balanced set used as the GA fitness signal
      3) Exhaustive warm-up — a *separate* KMeans pass for the exhaustive test's
                              cluster-jump attackers (never seen during GA warm-up)
      4) Exhaustive eval    — held-out set for the final exhaustive C(N,m) test
    """
    print("=== Preprocessing (runs once, shared across all sweep experiments) ===")

    eval_idx_set = set(get_eval_indices(dataset, N_PER_CLASS, seed=42))
    pool = [i for i in range(len(dataset)) if i not in eval_idx_set]
    _random.Random(42).shuffle(pool)
    n_pre = N_PREPROCESS_BATCHES * BATCH_SIZE

    if len(pool) < 2 * n_pre:
        raise ValueError(
            f"Not enough images ({len(pool)}) to carve out two disjoint "
            f"{n_pre}-image warm-up sets plus a held-out eval set."
        )

    ga_warmup_pool     = pool[:n_pre]
    final_warmup_pool  = pool[n_pre:2 * n_pre]
    final_eval_indices = pool[2 * n_pre:]

    _g = torch.Generator()
    _g.manual_seed(42)
    ga_pre_loader = DataLoader(
        Subset(dataset, ga_warmup_pool),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, generator=_g,
    )

    _g2 = torch.Generator()
    _g2.manual_seed(43)
    final_pre_loader = DataLoader(
        Subset(dataset, final_warmup_pool),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, generator=_g2,
    )

    final_eval_loader = DataLoader(
        Subset(dataset, final_eval_indices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    mini_eval_loader = make_balanced_mini_loader(dataset, N_PER_CLASS, BATCH_SIZE, seed=42)
    print(f"GA mini-eval:        {len(mini_eval_loader.dataset):>6} images (GA fitness signal)")
    print(f"GA warm-up:          {len(ga_warmup_pool):>6} images (GA attacker KMeans warm-up)")
    print(f"Exhaustive warm-up:  {len(final_warmup_pool):>6} images (exhaustive attacker KMeans warm-up)")
    print(f"Exhaustive eval:     {len(final_eval_indices):>6} images (held-out, disjoint from all of the above)\n")

    baseline_specs = client_specs_from_blueprint(devices, DeviceBlueprint.from_dict({}), device_ids)
    preprocess_topology = TopologySpec(clients=[
        ClientSpec(
            client_id=cs.client_id, layer_range=cs.layer_range,
            height_range=cs.height_range, replicas=[CleanAttack()],
            replica_device_ids=cs.replica_device_ids,
        )
        for cs in baseline_specs
    ])

    def _build_centroids(loader):
        preprocess_model = MicrosplitModel(
            flat_model, preprocess_topology, get_layer_fn, set_layer_fn,
        ).to(device)
        preprocess_clustering(
            preprocess_model, loader,
            n_batches=N_PREPROCESS_BATCHES, n_clusters=N_CLUSTERS, device=device,
        )
        result = extract_centroids(preprocess_model)
        del preprocess_model
        return result

    print("--- GA attacker warm-up ---")
    centroids = _build_centroids(ga_pre_loader)
    print(f"Centroids ready for: {list(centroids.keys())}\n")

    print("--- Exhaustive-test attacker warm-up ---")
    final_centroids = _build_centroids(final_pre_loader)
    print(f"Centroids ready for: {list(final_centroids.keys())}\n")

    return centroids, mini_eval_loader, final_centroids, final_eval_loader
