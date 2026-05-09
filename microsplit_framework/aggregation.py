from __future__ import annotations
import torch
from abc import ABC, abstractmethod


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, replica_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Receives one tensor per replica, all the same shape (the client's height slice).
        Returns a single tensor of the same shape.
        """
        pass


class MedianAggregation(AggregationStrategy):
    """
    Element-wise median across replicas. Robust to a minority of malicious replicas
    and the natural continuous generalisation of majority voting.
    """

    def aggregate(self, replica_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(replica_outputs) == 1:
            return replica_outputs[0]
        stacked = torch.stack(replica_outputs, dim=0)
        return torch.median(stacked, dim=0).values


class MeanAggregation(AggregationStrategy):
    def aggregate(self, replica_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(replica_outputs) == 1:
            return replica_outputs[0]
        stacked = torch.stack(replica_outputs, dim=0)
        return torch.mean(stacked, dim=0)


class DetectionMeanAggregation(AggregationStrategy):
    """
    Compares every replica to the first replica using exact equality.
    Counts how many forward passes have at least one mismatching replica.
    Returns the mean so inference continues regardless of detection.
    """

    def __init__(self, client_id: str = "unknown"):
        self.client_id     = client_id
        self.total_count:    int = 0
        self.mismatch_count: int = 0

    @property
    def detection_rate(self) -> float:
        return self.mismatch_count / self.total_count if self.total_count > 0 else 0.0

    def aggregate(self, replica_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(replica_outputs) == 1:
            return replica_outputs[0]

        self.total_count += 1
        reference = replica_outputs[0]
        mismatch = any(
            not torch.equal(output, reference)
            for output in replica_outputs[1:]
        )
        if mismatch:
            self.mismatch_count += 1

        stacked = torch.stack(replica_outputs, dim=0)
        return torch.mean(stacked, dim=0)
