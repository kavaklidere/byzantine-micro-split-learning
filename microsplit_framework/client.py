from __future__ import annotations
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

from microsplit_framework.attacks import AttackConfig, ClusterJumpAttack
from microsplit_framework.aggregation import AggregationStrategy
from microsplit_framework.topology import resolve_height_pixels


class ReplicaModule(nn.Module):
    """
    One physical replica of a client. Holds an nn.Sequential spanning the
    client's full layer range. forward_segment runs only the sub-layers
    corresponding to the requested segment, so the same module is reused
    across all segments the client participates in.
    """

    def __init__(self, attack_config: AttackConfig, layers: nn.Sequential):
        super().__init__()
        self.attack_config = attack_config
        self.layers        = layers

    def forward_segment(
        self,
        x: torch.Tensor,
        seg_start: int,
        seg_end: int,
        layer_start: int,
    ) -> torch.Tensor:
        offset     = seg_start - layer_start
        seg_length = seg_end - seg_start + 1
        h = x
        for i in range(offset, offset + seg_length):
            h = self.layers[i](h)
        return h


class ClientModule(nn.Module):
    """
    Persistent per-client entity. One instance is created at build time and
    referenced by every SegmentRouter that covers any part of this client's
    layer range.

    Cluster vocabulary (eavesdrop_memory, cluster_centroids) lives here rather
    than on AttackConfig, because it belongs to the client's network position,
    not to any particular attack type. This allows any client to be switched to
    ClusterJumpAttack in a later experiment without re-running preprocessing.
    """

    def __init__(
        self,
        client_id: str,
        layer_start: int,
        layer_end: int,
        h_start_frac: float,
        h_end_frac: float,
        replicas: list[ReplicaModule],
        attack_configs: list[AttackConfig],
        aggregation: AggregationStrategy,
    ):
        super().__init__()
        self.client_id    = client_id
        self.layer_start  = layer_start
        self.layer_end    = layer_end
        self.h_start_frac = h_start_frac
        self.h_end_frac   = h_end_frac
        self.replicas      = nn.ModuleList(replicas)
        self.attack_configs = attack_configs   # plain list — AttackConfig is not nn.Module
        self.aggregation   = aggregation

        # Cluster vocabulary — built by preprocess_clustering(), used by ClusterJumpAttack
        self.is_cluster_ready:   bool                   = False
        self.eavesdrop_memory:   list[torch.Tensor]     = []
        self.cluster_centroids:  torch.Tensor | None    = None
        self._eavesdrop_shape:   tuple | None           = None   # shape of one slice (C,H,W) or (F,)

    def eavesdrop(self, activation_slice: torch.Tensor) -> None:
        if self._eavesdrop_shape is None:
            self._eavesdrop_shape = tuple(activation_slice.shape[1:])
        self.eavesdrop_memory.append(activation_slice.detach().cpu())

    def build_clusters(self, device: torch.device, n_clusters: int = 100) -> None:
        all_data = torch.cat(self.eavesdrop_memory, dim=0)
        flat     = all_data.view(all_data.size(0), -1).cpu().numpy()

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, n_init="auto", random_state=42)
        kmeans.fit(flat)

        original_shape = all_data.shape[1:]
        self.cluster_centroids = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        ).view(-1, *original_shape)

        self.eavesdrop_memory.clear()
        self.is_cluster_ready = True

    def get_malicious_slice(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n = self.cluster_centroids.size(0)
        indices = torch.randint(0, n, (batch_size,), device=device)
        return self.cluster_centroids[indices]

    def forward_segment(
        self,
        x: torch.Tensor,
        seg_start: int,
        seg_end: int,
    ) -> torch.Tensor:
        """
        Called by SegmentRouter for every segment this client participates in.
        Returns the aggregated height-slice tensor ready to be patched into
        the assembled output.
        """
        is_final = (seg_end == self.layer_end)

        # Run all replicas first — pixel indices must be computed from the
        # OUTPUT shape, not the input shape, because pooling layers change height.
        full_outputs: list[torch.Tensor] = [
            replica.forward_segment(x, seg_start, seg_end, self.layer_start)
            for replica in self.replicas
        ]

        first_out  = full_outputs[0]
        is_spatial = (first_out.dim() == 4)
        total_dim  = first_out.shape[2] if is_spatial else first_out.shape[1]

        h_start_px, h_end_px = resolve_height_pixels(
            self.h_start_frac, self.h_end_frac, total_dim
        )

        slices: list[torch.Tensor] = []

        for idx, (full_output, attack_config) in enumerate(
            zip(full_outputs, self.attack_configs)
        ):
            if is_spatial:
                clean_slice = full_output[:, :, h_start_px:h_end_px, :]
            else:
                clean_slice = full_output[:, h_start_px:h_end_px]

            # Eavesdrop using the first replica's output (once per client per batch)
            if is_final and not self.is_cluster_ready and idx == 0:
                self.eavesdrop(clean_slice)

            if isinstance(attack_config, ClusterJumpAttack) and is_final and self.is_cluster_ready:
                effective_slice = self.get_malicious_slice(clean_slice.size(0), full_output.device)
            else:
                effective_slice = clean_slice

            slices.append(effective_slice)

        return self.aggregation.aggregate(slices)
