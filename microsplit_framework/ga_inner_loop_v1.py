from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from microsplit_framework.topology import ClientSpec, TopologySpec, DeviceSpec
from microsplit_framework.attacks import CleanAttack, ClusterJumpAttack
from microsplit_framework.aggregation import AggregationStrategy, MedianAggregation
from microsplit_framework.model import MicrosplitModel


@dataclass(frozen=True)
class RedundancyBlueprint:
    """
    Specifies the number of replicas per client. Hashable — safe as a GA dict key.
    Clients not listed default to 1 replica.
    """
    _counts: frozenset

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> "RedundancyBlueprint":
        return cls(frozenset(d.items()))

    def n_replicas(self, client_id: str) -> int:
        return 1 + {k: v for k, v in self._counts}.get(client_id, 0)

    def total_replicas(self, all_client_ids: list[str]) -> int:
        return sum(self.n_replicas(cid) for cid in all_client_ids)

    def __repr__(self) -> str:
        return f"Blueprint({dict(sorted(self._counts))})"


@dataclass(frozen=True)
class DeviceBlueprint:
    """
    Specifies extra replicas per (partition, device) pair. Hashable — safe as a GA dict key.
    budget K = sum of all extra counts.
    e.g. {("conv2", "device_1"): 1, ("conv2", "device_3"): 1} means conv2 gets 1 extra on
    device_1 and 1 extra on device_3, regardless of which device owns the conv2 baseline.
    """
    _assignments: frozenset  # frozenset of (partition_id, device_id, count)

    @classmethod
    def from_dict(cls, d: dict[tuple[str, str], int]) -> "DeviceBlueprint":
        return cls(frozenset((p, dv, c) for (p, dv), c in d.items() if c > 0))

    def n_extras(self, partition_id: str, device_id: str) -> int:
        for p, dv, c in self._assignments:
            if p == partition_id and dv == device_id:
                return c
        return 0

    def all_assignments(self) -> dict[tuple[str, str], int]:
        return {(p, dv): c for p, dv, c in self._assignments}

    def total_extras(self) -> int:
        return sum(c for _, _, c in self._assignments)

    def __repr__(self) -> str:
        assigns = ", ".join(f"({p},{dv}):{c}" for p, dv, c in sorted(self._assignments))
        return f"DeviceBP({{{assigns}}})"


def client_specs_from_blueprint(
    devices: list[DeviceSpec],
    blueprint: "DeviceBlueprint",
    device_ids: list[str],
) -> list[ClientSpec]:
    """
    Build a flat list of ClientSpecs for a given DeviceBlueprint.

    Each partition gets one baseline replica (owned by its DeviceSpec's device_id)
    plus any extra replicas assigned to other devices by the blueprint.
    The replica_device_ids field records which device owns each replica position.
    """
    result = []
    for base_device in devices:
        for cs in base_device.client_specs:
            pid = cs.client_id
            replica_dev_ids = [base_device.device_id]
            for dev_id in device_ids:
                n_extra = blueprint.n_extras(pid, dev_id)
                replica_dev_ids.extend([dev_id] * n_extra)
            result.append(ClientSpec(
                client_id=pid,
                layer_range=cs.layer_range,
                height_range=cs.height_range,
                replicas=[CleanAttack()] * len(replica_dev_ids),
                aggregation=cs.aggregation,
                replica_device_ids=replica_dev_ids,
            ))
    return result


@dataclass
class InnerLoopResult:
    blueprint:    RedundancyBlueprint
    m_attackers:  int
    min_top1:     float
    min_top5:     float
    worst_combo:  frozenset
    per_combo:    dict = field(default_factory=dict)


def extract_centroids(model: nn.Module) -> dict[str, torch.Tensor | None]:
    """Snapshot cluster centroids from a preprocessed model."""
    return {cm.client_id: cm.cluster_centroids for cm in model.client_modules}


def inject_centroids(model: nn.Module, centroids: dict[str, torch.Tensor | None]) -> None:
    """Copy pre-computed centroids into a freshly-built model."""
    for cm in model.client_modules:
        tensor = centroids.get(cm.client_id)
        if tensor is not None:
            cm.cluster_centroids = tensor
            cm.is_cluster_ready = True


def get_eval_indices(dataset, n_per_class: int = 1, seed: int = 42) -> list[int]:
    """
    Randomly select n_per_class images per class using a fixed seed.
    Returns a reproducible eval split that covers all classes regardless of
    how the dataset is ordered on disk.
    """
    import random as _random
    rng = _random.Random(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)
    selected: list[int] = []
    for cls_indices in class_to_indices.values():
        chosen = rng.sample(cls_indices, min(n_per_class, len(cls_indices)))
        selected.extend(chosen)
    return selected


def make_balanced_mini_loader(
    dataset,
    n_per_class: int = 1,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    """
    Build an eval loader with n_per_class randomly chosen images per class.
    The seed makes the split reproducible. Call get_eval_indices() with the
    same seed to find which indices are reserved so the preprocessing loader
    can exclude them.
    """
    selected = get_eval_indices(dataset, n_per_class, seed)
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )


def build_topology_for_combo(
    base_specs: list[ClientSpec],
    blueprint: RedundancyBlueprint,
    malicious_replicas: frozenset,
) -> TopologySpec:
    """
    Build a TopologySpec for one attacker combination.

    malicious_replicas: frozenset of (client_id, replica_idx) tuples that
    use ClusterJumpAttack; all others use CleanAttack.
    """
    clients: list[ClientSpec] = []
    for spec in base_specs:
        cid = spec.client_id
        n = blueprint.n_replicas(cid)
        replicas = [
            ClusterJumpAttack() if (cid, i) in malicious_replicas else CleanAttack()
            for i in range(n)
        ]
        clients.append(ClientSpec(
            client_id    = cid,
            layer_range  = spec.layer_range,
            height_range = spec.height_range,
            replicas     = replicas,
            aggregation  = MedianAggregation(),
        ))
    return TopologySpec(clients=clients)


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


def evaluate_inner_loop(
    base_model: nn.Module,
    devices: list[DeviceSpec],
    blueprint: "DeviceBlueprint",
    m_attackers: int,
    centroids: dict[str, torch.Tensor | None],
    mini_eval_loader: DataLoader,
    get_layer_fn: Callable,
    set_layer_fn: Callable | None,
    device: torch.device,
    aggregation: AggregationStrategy | None = None,
    verbose: bool = False,
) -> InnerLoopResult:
    """
    Exhaustively test all C(N, m) device-combo attacker scenarios for a given blueprint.

    N = number of devices.  m = m_attackers (compromised devices per combo).
    The model is built ONCE. Between combos, attack_configs are swapped in-place
    for all replicas owned by the compromised devices — no weight copies needed.
    Coordinated centroid sharing (same device → same centroid) is handled inside
    ClientModule.forward_segment via replica_device_ids.
    """
    device_ids = [d.device_id for d in devices]
    client_specs = client_specs_from_blueprint(devices, blueprint, device_ids)

    n_total  = len(device_ids)
    n_combos = math.comb(n_total, m_attackers)
    print(
        f"[InnerLoop] Blueprint: {blueprint}  |  m={m_attackers}  |  "
        f"N={n_total} devices  |  C(N,m)={n_combos} combos"
    )

    agg = aggregation or MedianAggregation()

    # Build the model ONCE — all replicas start as CleanAttack with device ownership
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

    # device_id → list of (ClientModule, replica_idx) — all replicas owned by that device
    device_to_replicas: dict[str, list[tuple]] = {}
    for cm in model.client_modules:
        for i, dev_id in enumerate(cm.replica_device_ids):
            device_to_replicas.setdefault(dev_id, []).append((cm, i))

    _clean     = CleanAttack()
    _malicious = ClusterJumpAttack()

    per_combo: dict[frozenset, tuple[float, float]] = {}
    min_top1    = float("inf")
    min_top5    = float("inf")
    worst_combo: frozenset = frozenset()

    for combo in itertools.combinations(device_ids, m_attackers):
        # Swap all replicas owned by the compromised devices to malicious
        for dev_id in combo:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _malicious

        top1, top5 = _evaluate_mini_batch(model, mini_eval_loader, device)

        # Restore to clean before next combo
        for dev_id in combo:
            for cm, i in device_to_replicas.get(dev_id, []):
                cm.attack_configs[i] = _clean

        malicious_set = frozenset(combo)
        per_combo[malicious_set] = (top1, top5)

        if verbose:
            print(f"  Attackers: [{', '.join(sorted(combo))}]  ->  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

        if top1 < min_top1:
            min_top1    = top1
            min_top5    = top5
            worst_combo = malicious_set

    del model

    return InnerLoopResult(
        blueprint   = blueprint,
        m_attackers = m_attackers,
        min_top1    = min_top1,
        min_top5    = min_top5,
        worst_combo = worst_combo,
        per_combo   = per_combo,
    )
