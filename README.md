# MrNEAD

**MrNEAD: Community Detection Attacks in Social Networks Using Modularity-Regularized Network Embedding via Adversarial Decomposition**

## Algorithm Overview

MrNEAD optimizes the following objective function:

```
min_{H,M,H_m} ||A-HH^T||_F^2 + α||A+M-H_m H_m^T||_F^2 + β Tr(HH^T H_m H_m^T) + γ||M-Q̂||_F^2
```

Subject to:
- H ≥ 0, H_m ≥ 0 (non-negativity constraints)
- A+M ∈ [0,1] (valid adjacency matrix)
- ||M||_0 < ξ (sparsity constraint)

Where:
- **H**: Original network embedding (n×k)
- **H_m**: Attacked network embedding (n×k) 
- **M**: Adversarial modification matrix (n×n)
- **Q̂**: Inverted modularity matrix
- **α, β, γ**: Regularization parameters
- **ξ**: Attack budget (number of edge modifications)

## Components

### NEAD (Network Embedding Attack and Defense)
- `||A-HH^T||_F^2`: Reconstruction error for original network
- `α||A+M-H_m H_m^T||_F^2`: Reconstruction error for attacked network
- `β Tr(HH^T H_m H_m^T)`: Adversarial perturbation term

### MIGS (Modularity-Inverted Guidance Strategy)
- `γ||M-Q̂||_F^2`: Guides attacks using inverted modularity matrix

## Files Structure

```
demo_MrNEAD/
├── main.py              # Main demonstration program
├── mrNEAD_core.py       # Core MrNEAD algorithm implementation
├── utils.py             # Utility functions (data loading, evaluation)
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── data/                # Network data directory

```


## Parameters

- **k**: Embedding dimension
- **ξ**: Attack budget
- **α**: Reconstruction weight
- **β**: Modularity guidance weight
- **γ**: Adversarial weight
- **max_iter**: Maximum iterations

### Evaluation Metrics
- **NMI**: Normalized Mutual Information between original and attacked communities (lower is better for attacks)
- **ARI**: Adjusted Rand Index between original and attacked communities (lower is better for attacks)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- networkx
- scikit-learn
- scipy
- python-louvain (community)

