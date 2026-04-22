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
  하이퍼엣지 : ~1100–1300  (research-group 기준값에 따라 다름)
  클래스 수  : 7
  Train/Val/Test : 140 / 500 / 1000

=== Laplacian 계산 완료 ===

=== 학습 시작 ===
Epoch  10 | Loss: x.xxxx | Train Acc: x.xxxx | Val Acc: x.xxxx
...
Epoch 200 | Loss: x.xxxx | Train Acc: x.xxxx | Val Acc: x.xxxx
=== 학습 완료 ===

=== 최종 테스트 정확도: x.xxxx ===
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

### Hyperedge Construction (Cora)

Cora의 하이퍼엣지는 **자주 인용되는 논문 중심의 research-group** 방식으로 구성됩니다:

```
논문 A가 5회 이상 인용됨
논문 A를 인용하는 논문들: {B, C, D, E, ...}
   → 연구 그룹 하이퍼엣지 = {A, B, C, D, E, ...}
```

기본 설정에서는 피인용 횟수가 `min_group_citations = 5` 이상인 논문을 연구 그룹의 seed로 사용합니다. 즉, 많이 인용되는 핵심 논문과 그 논문을 인용한 논문들을 하나의 하이퍼엣지로 묶어, 같은 연구 흐름을 공유하는 논문 그룹을 고차(higher-order) 관계로 표현합니다.

기준값은 `main.m`에서 조정할 수 있습니다:

```matlab
data_options.min_group_citations = 5;
data_options.include_singletons  = true;
```

> **Note:** 어떤 연구 그룹 하이퍼엣지에도 속하지 않는 노드에는 싱글턴 하이퍼엣지가 자동으로 추가됩니다.


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
