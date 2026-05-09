from __future__ import annotations
import copy
from typing import Callable

import torch
import torch.nn as nn

from microsplit_framework.topology import TopologySpec, ClientSpec
from microsplit_framework.attacks import AttackConfig
from microsplit_framework.aggregation import AggregationStrategy, MedianAggregation
from microsplit_framework.client import ClientModule, ReplicaModule
from microsplit_framework.segment import SegmentRouter


class MicrosplitModel(nn.Module):
    """
    Top-level nn.Module. Derives segments from the topology, builds one
    ClientModule per logical client, wraps each segment in a SegmentRouter,
    and either:
      - (full-model mode) injects segment routers into a deep copy of the
        original model so forward() returns the model's complete output, or
      - (topology-only mode) chains only the segment routers, returning the
        output of the last covered layer.

    Parameters
    ----------
    original_model : nn.Module
        The pretrained model to partition. Its weights are never mutated —
        every client and clean-path copy uses deep copies.
    topology : TopologySpec
        Declarative description of all clients, their layer/height ranges,
        and their replica attack configurations.
    get_layer_fn : Callable[[nn.Module, int], nn.Module]
        Maps a layer index to the corresponding layer in original_model.
        Example for VGG16: ``lambda m, i: m.features[i]``
    set_layer_fn : Callable[[nn.Module, int, nn.Module], None] | None
        Maps a layer index to a setter in the model copy. When provided,
        segment routers are injected into a full deep copy of original_model
        so that forward() returns the complete model output (e.g. logits).
        Example for VGG16: ``lambda m, i, v: m.features.__setitem__(i, v)``
        When None, forward() returns only the output of the topology range.
    aggregation_factory : Callable[[ClientSpec], AggregationStrategy] | None
        Called once per ClientSpec to produce the aggregation strategy for
        that client's replicas. Defaults to MedianAggregation for all clients.
    """

    def __init__(
        self,
        original_model: nn.Module,
        topology: TopologySpec,
        get_layer_fn: Callable[[nn.Module, int], nn.Module],
        set_layer_fn: Callable[[nn.Module, int, nn.Module], None] | None = None,
        aggregation_factory: Callable[[ClientSpec], AggregationStrategy] | None = None,
    ):
        super().__init__()

        if aggregation_factory is None:
            aggregation_factory = lambda spec: MedianAggregation()

        segments   = topology.derive_segments()
        client_map: dict[str, ClientModule] = {}

        for spec in topology.clients:
            replica_modules: list[ReplicaModule] = []
            isolated_configs: list[AttackConfig] = []

            for attack_config in spec.replicas:
                isolated = copy.deepcopy(attack_config)
                isolated_configs.append(isolated)

                raw_layers = [
                    copy.deepcopy(get_layer_fn(original_model, i))
                    for i in range(spec.layer_range[0], spec.layer_range[1] + 1)
                ]
                clean_seq  = nn.Sequential(*raw_layers)
                built_seq  = isolated.build_layers(clean_seq)
                replica_modules.append(ReplicaModule(attack_config=isolated, layers=built_seq))

            client_map[spec.client_id] = ClientModule(
                client_id      = spec.client_id,
                layer_start    = spec.layer_range[0],
                layer_end      = spec.layer_range[1],
                h_start_frac   = spec.height_range[0],
                h_end_frac     = spec.height_range[1],
                replicas       = replica_modules,
                attack_configs = isolated_configs,
                aggregation    = spec.aggregation if spec.aggregation is not None else aggregation_factory(spec),
            )

        seg_router_list: list[SegmentRouter] = []
        for seg in segments:
            clean_layers_list = [
                copy.deepcopy(get_layer_fn(original_model, i))
                for i in range(seg.seg_start, seg.seg_end + 1)
            ]
            clean_seq      = nn.Sequential(*clean_layers_list)
            active_clients = [client_map[c.client_id] for c in seg.clients]
            seg_router_list.append(SegmentRouter(
                seg_start      = seg.seg_start,
                seg_end        = seg.seg_end,
                clean_layers   = clean_seq,
                client_modules = active_clients,
            ))

        # Always register segment routers and client modules so their parameters
        # are reachable via .parameters() and .to(device) in both modes.
        self.segment_routers = nn.ModuleList(seg_router_list)
        self.client_modules  = nn.ModuleList(list(client_map.values()))

        if set_layer_fn is not None:
            # Full-model mode: inject routers into a complete deep copy of the
            # original model. Layers inside each segment range that come after
            # the router entry point are replaced with Identity so the router's
            # internal clean path handles them.
            self._full_model = copy.deepcopy(original_model)
            for seg, router in zip(segments, seg_router_list):
                set_layer_fn(self._full_model, seg.seg_start, router)
                for i in range(seg.seg_start + 1, seg.seg_end + 1):
                    set_layer_fn(self._full_model, i, nn.Identity())
        else:
            self._full_model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._full_model is not None:
            return self._full_model(x)

        h = x
        for router in self.segment_routers:
            h = router(h)
        return h
