# Byzantine Micro-Split Learning
Distributed micro-split neural networks allow for flexible, both vertical and horizontal, partitioning of the underlying deep neural network among multiple clients. This partitioning introduces a critical vulnerability, allowing Byzantine clients to arbitrarily corrupt their assigned splits. In this paper, we propose ClusterJump, an attack based on substituting honest activations with authentic but malicious representations drawn from an adversarially learned latent vocabulary. To counteract this threat, we propose an algorithm to discover optimal, budget-constrained redundancy placements and restore Byzantine Fault Tolerance. 

## Installation


## Environment
This framework was developed and tested using the following environment:
* Python: 3.12.3
* PyTorch: 2.11.0+cu130
* torchvision: 0.26.0+cu130
* scikit-learn: 1.8.0
* CUDA: 13.0, Driver 580.126.16
* GPU: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
