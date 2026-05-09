from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AttackConfig(ABC):
    @abstractmethod
    def build_layers(self, clean_seq: nn.Sequential) -> nn.Sequential:
        """
        Receives a deep-copied nn.Sequential of the client's full layer range.
        Returns the nn.Sequential this replica will use for its forward pass.
        May mutate clean_seq in-place (e.g. NoiseInjection) or return it unchanged.
        """
        pass

    @abstractmethod
    def is_malicious(self) -> bool:
        pass


class CleanAttack(AttackConfig):
    def build_layers(self, clean_seq: nn.Sequential) -> nn.Sequential:
        return clean_seq

    def is_malicious(self) -> bool:
        return False


class NoiseInjectionAttack(AttackConfig):
    def __init__(self, std_dev: float = 0.05):
        self.std_dev = std_dev

    def build_layers(self, clean_seq: nn.Sequential) -> nn.Sequential:
        with torch.no_grad():
            for module in clean_seq.modules():
                if hasattr(module, "weight") and module.weight is not None:
                    noise = torch.randn_like(module.weight) * self.std_dev
                    module.weight.add_(noise)
        return clean_seq

    def is_malicious(self) -> bool:
        return True


class ClusterJumpAttack(AttackConfig):
    """
    Marker attack: signals that this replica should replace its height slice with
    a random cluster centroid drawn from the client's pre-built vocabulary.
    Cluster state lives on ClientModule; this class carries no state of its own.
    Call preprocess_clustering() before evaluation to build the vocabulary.
    """

    def build_layers(self, clean_seq: nn.Sequential) -> nn.Sequential:
        return clean_seq

    def is_malicious(self) -> bool:
        return True
