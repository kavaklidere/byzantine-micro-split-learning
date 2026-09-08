from __future__ import annotations
import math

import torch
import torch.nn as nn

from microsplit_framework.topology import ClientSpec


def _layer_type(layer: nn.Module) -> str:
    if isinstance(layer, nn.Conv2d):
        return "conv"
    if isinstance(layer, nn.Linear):
        return "dense"
    if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return "pool"
    if isinstance(layer, nn.Flatten):
        return "flatten"
    return "act"  # ReLU, Dropout, BatchNorm, etc.


def _kernel_numel(layer: nn.Conv2d) -> int:
    k = layer.kernel_size
    if isinstance(k, int):
        return k * k
    return k[0] * k[1]


def profile_layer_shapes(
    model: nn.Sequential,
    input_shape: tuple = (1, 3, 224, 224),
) -> dict[int, tuple[tuple, tuple]]:
    """
    Profile input/output shapes for each layer in a flat nn.Sequential.
    Returns {layer_idx: (in_shape, out_shape)} with batch dim excluded.

    Asserts that every layer was captured exactly once — catches reused module
    instances (shared weights) where a hook would overwrite earlier captures.
    """
    captured: dict[int, tuple[tuple, tuple]] = {}
    hooks = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            captured[idx] = (tuple(inp[0].shape[1:]), tuple(out.shape[1:]))
        return hook

    for i, layer in enumerate(model):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(torch.zeros(*input_shape))

    for h in hooks:
        h.remove()

    assert len(captured) == len(model), (
        f"profile_layer_shapes: captured {len(captured)} layers but model has "
        f"{len(model)}. A module instance may be reused (shared weights)."
    )
    return captured


def compute_partition_memory(
    model: nn.Sequential,
    spec: ClientSpec,
    layer_shapes: dict[int, tuple[tuple, tuple]],
) -> dict:
    """
    Compute weight and activation-peak element counts for one ClientSpec.

    Uses math.ceil for all owned-extent calculations — conservative for the
    hard capacity constraint (never understates cost, never gives a false
    "feasible" verdict).

    Returns:
        {
            "weight_elements":          W_i  (int),
            "activation_peak_elements": P_i  (int),
            "total_elements":           W_i + P_i  (int),
        }
    """
    l_start, l_end = spec.layer_range
    gamma = spec.height_range[1] - spec.height_range[0]

    weight_elements = 0
    peak_act_elements = 0

    for l in range(l_start, l_end + 1):
        layer = model[l]
        lt = _layer_type(layer)
        in_shape, out_shape = layer_shapes[l]

        # ── Weight cost ──────────────────────────────────────────────────────
        if lt == "conv":
            # Full filter bank — independent of spatial slice (γ-invariant)
            weight_elements += layer.out_channels * layer.in_channels * _kernel_numel(layer)
        elif lt == "dense":
            # Only the ceil(γ × N_out) owned output rows
            weight_elements += math.ceil(gamma * layer.out_features) * layer.in_features
        # pool, flatten, act: 0

        # ── Activation peak m_l (dispatch on layer type, NOT on ndim) ────────
        # Never dispatch on out_shape ndim: flatten has in 3D, out 1D, so
        # reading out_shape ndim would silently misroute it to the 1-D branch.
        if lt in ("conv", "pool"):
            C_in, H_in, W_in = in_shape
            C_out, H_out, W_out = out_shape
            m_l = (C_in * math.ceil(gamma * H_in) * W_in
                   + C_out * math.ceil(gamma * H_out) * W_out)
        elif lt == "dense":
            # Dense needs the full input vector; output is the owned slice
            m_l = math.prod(in_shape) + math.ceil(gamma * out_shape[0])
        elif lt == "flatten":
            # Reshape aliases the input buffer — count input slice only
            C, H, W = in_shape  # always 3D for a flatten layer
            m_l = C * math.ceil(gamma * H) * W
        else:  # act: ReLU, Dropout, etc. — in-place, no new allocation
            if len(in_shape) == 3:
                C, H, W = in_shape
                m_l = C * math.ceil(gamma * H) * W
            else:
                m_l = math.ceil(gamma * in_shape[0])

        peak_act_elements = max(peak_act_elements, m_l)

    return {
        "weight_elements":          weight_elements,
        "activation_peak_elements": peak_act_elements,
        "total_elements":           weight_elements + peak_act_elements,
    }


def compute_memory_costs(
    model: nn.Sequential,
    client_specs: list[ClientSpec],
    input_shape: tuple = (1, 3, 224, 224),
) -> dict[str, dict]:
    """
    Profile the model once and return memory cost dicts keyed by client_id.

    Call once per topology (before the GA). Store the result and pass it into
    the minimax evaluation wherever device capacity constraints are needed.
    """
    layer_shapes = profile_layer_shapes(model, input_shape)

    # Confirm every spec's layer indices exist in the profiled dict — catches
    # mismatches between hand-written layer_range values and enumerate(model).
    for spec in client_specs:
        for l in range(spec.layer_range[0], spec.layer_range[1] + 1):
            assert l in layer_shapes, (
                f"Layer index {l} from spec '{spec.client_id}' not found in profiled "
                f"layer_shapes (model has {len(layer_shapes)} layers, "
                f"indexed 0–{len(layer_shapes) - 1}). "
                f"Check that ClientSpec.layer_range uses the same enumeration as "
                f"enumerate(model)."
            )

    return {
        spec.client_id: compute_partition_memory(model, spec, layer_shapes)
        for spec in client_specs
    }


def validate_tiling(
    model: nn.Sequential,
    client_specs: list[ClientSpec],
    layer_shapes: dict[int, tuple[tuple, tuple]],
) -> None:
    """
    Debug pass: for each spatial/dense layer, sum the ceil(γ_i × size)
    owned extents across all partitions that include it and compare to actual
    layer size. Logs the drift per layer.

    Expected drift with ceil: 0 (clean split) to +n_partitions (each
    partition rounded up by ≤1). A drift larger than that, or any negative
    drift (undercount), signals a real bug — not just rounding.

    Not exported; call explicitly during debugging.
    """
    layer_to_specs: dict[int, list[tuple[ClientSpec, float]]] = {}
    for spec in client_specs:
        gamma = spec.height_range[1] - spec.height_range[0]
        for l in range(spec.layer_range[0], spec.layer_range[1] + 1):
            layer_to_specs.setdefault(l, []).append((spec, gamma))

    print("[validate_tiling]")
    for l in sorted(layer_to_specs.keys()):
        in_shape, out_shape = layer_shapes[l]
        specs_at_layer = layer_to_specs[l]
        layer = model[l]
        lt = _layer_type(layer)

        if lt in ("conv", "pool") and len(out_shape) == 3:
            actual_H = out_shape[1]
            tiling_sum = sum(math.ceil(g * actual_H) for _, g in specs_at_layer)
            drift = tiling_sum - actual_H
            n = len(specs_at_layer)
            status = (
                "OK" if drift == 0
                else f"DRIFT (+{drift}, known ceil rounding, ≤{n} expected)" if 0 < drift <= n
                else f"BUG: drift={drift} exceeds expected envelope" if drift > n
                else f"BUG: undercount drift={drift}"
            )
            print(f"  Layer {l:>3} ({lt:>8}, H={actual_H:>4}): tiling_sum={tiling_sum:>5} — {status}")

        elif lt == "dense" and len(out_shape) == 1:
            actual_N = out_shape[0]
            tiling_sum = sum(math.ceil(g * actual_N) for _, g in specs_at_layer)
            drift = tiling_sum - actual_N
            n = len(specs_at_layer)
            status = (
                "OK" if drift == 0
                else f"DRIFT (+{drift}, known ceil rounding, ≤{n} expected)" if 0 < drift <= n
                else f"BUG: drift={drift} exceeds expected envelope" if drift > n
                else f"BUG: undercount drift={drift}"
            )
            print(f"  Layer {l:>3} ({lt:>8}, N={actual_N:>5}): tiling_sum={tiling_sum:>6} — {status}")
