from __future__ import annotations
import torch
import torch.nn as nn

from microsplit_framework.client import ClientModule
from microsplit_framework.topology import resolve_height_pixels


class SegmentRouter(nn.Module):
    """
    Handles the forward pass for one derived segment [seg_start, seg_end].

    Execution per forward call:
      1. Run clean_layers on the full input tensor.
      2. For each active client, call forward_segment to get its (possibly
         malicious) aggregated height slice.
      3. Patch each client's slice into a clone of the clean output.
      4. Return the assembled tensor as input to the next segment.

    client_modules is stored as a plain Python list (not nn.ModuleList) to
    avoid duplicate parameter registration for clients that span multiple
    segments. MicrosplitModel.client_modules owns all ClientModule parameters.
    """

    def __init__(
        self,
        seg_start: int,
        seg_end: int,
        clean_layers: nn.Sequential,
        client_modules: list[ClientModule],
    ):
        super().__init__()
        self.seg_start      = seg_start
        self.seg_end        = seg_end
        self.clean_layers   = clean_layers
        self.client_modules = client_modules   # plain list — see docstring

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean_output = x
        for layer in self.clean_layers:
            clean_output = layer(clean_output)

        assembled  = clean_output.clone()
        is_spatial = (assembled.dim() == 4)
        total_dim  = assembled.shape[2] if is_spatial else assembled.shape[1]

        for client in self.client_modules:
            agg_slice = client.forward_segment(x, self.seg_start, self.seg_end)

            h_start_px, h_end_px = resolve_height_pixels(
                client.h_start_frac, client.h_end_frac, total_dim
            )

            if is_spatial:
                assembled[:, :, h_start_px:h_end_px, :] = agg_slice
            else:
                assembled[:, h_start_px:h_end_px] = agg_slice

        return assembled
