# HGNN — Hypergraph Neural Network (MATLAB)

A from-scratch MATLAB implementation of **Hypergraph Neural Networks (HGNN)** for semi-supervised node classification, based on the formulation from [Feng et al., 2019](https://arxiv.org/abs/1809.09401).

> **Key idea:** Unlike standard GNNs that operate on pairwise edges, HGNN models higher-order relationships through *hyperedges* — edges that can connect any number of nodes simultaneously.

---

## Architecture

```
Input Features (X)
       │
       ▼
┌──────────────────────────┐
│  HGNN Layer 1            │
│  ReLU( Θ · X · W₁ )     │
│  (N×F_in) → (N×hidden)   │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  HGNN Layer 2            │
│  Softmax( Θ · H₁ · W₂ ) │
│  (N×hidden) → (N×C)      │
└──────────────────────────┘
       │
       ▼
  Predicted Labels
```

The **normalized propagation matrix** Θ is computed as:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^\top \cdot D_v^{-1/2}$$

where *H* is the incidence matrix, *W* is the diagonal hyperedge weight matrix, *D_v* and *D_e* are the node and hyperedge degree matrices, respectively.

---

## Project Structure

```
HGNN/
├── main.m                          # Entry point — runs the full pipeline
├── data/
│   ├── load_data.m                 # Dataset loader (toy / Cora / custom)
│   └── cora/                       # Cora citation dataset files
├── graph/
│   ├── build_incidence_matrix.m    # Constructs sparse incidence matrix H
│   └── compute_laplacian.m         # Computes normalized propagation matrix Θ
├── model/
│   ├── hgnn_forward.m              # 2-layer forward pass
│   ├── hgnn_backward.m             # Backpropagation (manual gradient computation)
│   ├── hgnn_layer.m                # Single HGNN layer (propagation + activation)
│   └── hgnn_loss.m                 # Cross-entropy loss with mask support
└── train/
    ├── train.m                     # Training loop with Adam optimizer
    ├── evaluate.m                  # Classification accuracy evaluation
    └── validate.m                  # Validation wrapper
```

---

## Quick Start

### Requirements

- **MATLAB** R2016b or later (uses `containers.Map`, sparse ops, `bsxfun`)
- No additional toolboxes required

### Run

Open MATLAB, navigate to the project root, and execute:

```matlab
>> main
```

This will:
1. Load the **Cora** citation dataset (2,708 nodes, 7 classes)
2. Build the incidence matrix and compute the Laplacian
3. Train the HGNN for 200 epochs with Adam optimizer
4. Report the final test accuracy

### Expected Output

```
=== 데이터 로드 완료 ===
  노드 수    : 2708
  피처 차원  : 1433
  하이퍼엣지 : 2708
  클래스 수  : 7
  Train/Val/Test : 140 / 500 / 1000

=== Laplacian 계산 완료 ===

=== 학습 시작 ===
Epoch  10 | Loss: 1.8432 | Train Acc: 0.6571 | Val Acc: 0.4120
...
Epoch 200 | Loss: 0.2145 | Train Acc: 0.9929 | Val Acc: 0.7680
=== 학습 완료 ===

=== 최종 테스트 정확도: 0.7650 ===
```

---

## Datasets

| Dataset | Nodes | Features | Classes | Description |
|---------|------:|:--------:|:-------:|-------------|
| `toy`   | 12    | 5        | 3       | Synthetic data for quick debugging |
| `cora`  | 2,708 | 1,433    | 7       | Citation network (papers as nodes) |
| `custom`| —     | —        | —       | Extend `load_data.m` for your own data |

To switch datasets, edit `main.m`:

```matlab
dataset = 'toy';   % 'toy' | 'cora' | 'custom'
```

---

## Hyperparameters

All hyperparameters are configured in `main.m`:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `lr` | 0.01 | Learning rate |
| `epochs` | 200 | Number of training epochs |
| `hidden_dim` | 64 | Hidden layer dimension |
| `weight_decay` | 5e-4 | L2 regularization coefficient |
| `print_every` | 10 | Logging interval (epochs) |

---

## Implementation Details

- **Weight Initialization:** Xavier initialization for stable gradient flow
- **Optimizer:** Adam (β₁=0.9, β₂=0.999, ε=1e-8)
- **Loss:** Cross-entropy with numerical stability (ε=1e-8)
- **Activation:** ReLU (hidden) → Softmax (output)
- **Sparse Operations:** Incidence matrix and Laplacian stored as MATLAB sparse matrices for memory efficiency
- **Data Split:** 20 labeled nodes per class for training, 500 for validation, 1,000 for testing (following the standard Cora split protocol)

---

## References

- Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). *Hypergraph Neural Networks.* AAAI 2019. [[Paper]](https://arxiv.org/abs/1809.09401)

---

## License

This project is for educational and research purposes.
