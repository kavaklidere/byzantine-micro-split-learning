# Byzantine Micro-Split Learning
Distributed micro-split neural networks allow for flexible, both vertical and horizontal, partitioning of the underlying deep neural network among multiple clients. This partitioning introduces a critical vulnerability, allowing Byzantine clients to arbitrarily corrupt their assigned splits. In this paper, we propose ClusterJump, an attack based on substituting honest activations with authentic but malicious representations drawn from an adversarially learned latent vocabulary. To counteract this threat, we propose an algorithm to discover optimal, budget-constrained redundancy placements and restore Byzantine Fault Tolerance. 

## Installation
Follow these step-by-step instructions to set up your development environment.

### 1. Clone the Repository
First, clone this repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/kavaklidere/byzantine-micro-split-learning.git
cd byzantine-micro-split-learning
```
### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage the dependencies. Create and activate a virtual environment:
```bash
python3 -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
### 3. Install the Dependencies
```bash
pip install -r requirements.txt
```
⚠️ Important: Please ensure you have installed a CUDA-enabled version of PyTorch to be able to leverage GPU acceleration.

## Environment
This framework was developed and tested using the following environment:
* Python: 3.12.3
* PyTorch: 2.11.0+cu130
* torchvision: 0.26.0+cu130
* scikit-learn: 1.8.0
* CUDA: 13.0, Driver 580.126.16
* GPU: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
