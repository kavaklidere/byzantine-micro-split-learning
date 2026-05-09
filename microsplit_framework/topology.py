from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from microsplit_framework.attacks import AttackConfig
    from microsplit_framework.aggregation import AggregationStrategy


@dataclass
class ClientSpec:
    client_id: str
    layer_range: tuple[int, int]
    height_range: tuple[float, float]
    replicas: list["AttackConfig"]
    aggregation: "AggregationStrategy | None" = None

    def __post_init__(self):
        l0, l1 = self.layer_range
        h0, h1 = self.height_range
        if not (0 <= l0 <= l1):
            raise ValueError(f"Client '{self.client_id}': layer_range {self.layer_range} is invalid.")
        if not (0.0 <= h0 < h1 <= 1.0):
            raise ValueError(f"Client '{self.client_id}': height_range {self.height_range} must satisfy 0.0 <= start < end <= 1.0.")
        if len(self.replicas) < 1:
            raise ValueError(f"Client '{self.client_id}' must have at least one replica.")


@dataclass
class SegmentSpec:
    seg_start: int
    seg_end: int
    clients: list[ClientSpec]


@dataclass
class TopologySpec:
    clients: list[ClientSpec]

    def derive_segments(self) -> list[SegmentSpec]:
        breakpoints: set[int] = set()
        for c in self.clients:
            breakpoints.add(c.layer_range[0])
            breakpoints.add(c.layer_range[1] + 1)

        sorted_bps = sorted(breakpoints)
        segments: list[SegmentSpec] = []

        for i in range(len(sorted_bps) - 1):
            seg_start = sorted_bps[i]
            seg_end   = sorted_bps[i + 1] - 1

            active = [
                c for c in self.clients
                if c.layer_range[0] <= seg_start and c.layer_range[1] >= seg_end
            ]

            if active:
                segments.append(SegmentSpec(seg_start=seg_start, seg_end=seg_end, clients=active))

        return segments


def resolve_height_pixels(h_start_frac: float, h_end_frac: float, total: int) -> tuple[int, int]:
    """
    Convert fractional height spec to pixel indices (end is exclusive).

    Both boundaries use the same rounding formula, so adjacent clients that
    share a boundary fraction always map to the same pixel index — guaranteeing
    no gaps and no overlaps even when total is not evenly divisible.
    """
    start_px = int(round(h_start_frac * total))
    end_px   = int(round(h_end_frac   * total))
    return start_px, end_px
