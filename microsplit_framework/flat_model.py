import torch.nn as nn


def make_flat_sequential(*parts: nn.Module) -> nn.Sequential:
    """
    Concatenate model containers into a single flat nn.Sequential.
    nn.Sequential parts are expanded; any other nn.Module is appended as-is.

    Example for VGG16:
        flat = make_flat_sequential(
            model.features,   # indices 0–30
            model.avgpool,    # index 31
            nn.Flatten(),     # index 32
            model.classifier, # indices 33–39
        )
    """
    layers = []
    for part in parts:
        if isinstance(part, nn.Sequential):
            layers.extend(list(part))
        else:
            layers.append(part)
    return nn.Sequential(*layers)
