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
2. Build the incidence matrix and compute the normalized propagation matrix
3. Train the HGNN for 200 epochs with Adam optimizer
4. Report the final test accuracy

### Expected Output

```
=== Data loaded ===
  Nodes      : 2708
  Features   : 1433
  Hyperedges : ~1100-1300  (depends on the research-group threshold)
  Classes    : 7
  Train/Val/Test : 140 / 500 / 1000

=== Propagation matrix computed ===

=== Training started ===
Epoch  10 | Loss: x.xxxx | Train Acc: x.xxxx | Val Acc: x.xxxx
...
Epoch 200 | Loss: x.xxxx | Train Acc: x.xxxx | Val Acc: x.xxxx
=== Training completed ===

=== Final test accuracy: x.xxxx ===
```

---

## How It Works

### 1. Entry Point

`HGNN/main.m` is the full pipeline driver. It adds the `data`, `graph`, `model`, and `train` folders to the MATLAB path, selects a dataset, configures Cora-specific hyperedge options, and then runs data loading, graph construction, training, and final testing.

The main outputs passed through the pipeline are:

| Variable | Shape | Meaning |
|----------|-------|---------|
| `X` | `N x F` | Node-feature matrix. For Cora, each row is a normalized bag-of-words feature vector for one paper. |
| `H` | `N x E` | Sparse incidence matrix. `H(i,e)=1` means node `i` belongs to hyperedge `e`. |
| `Y_true` | `N x C` | One-hot class labels. |
| `train_mask` | `N x 1` | Boolean mask for labeled training nodes. |
| `val_mask` | `N x 1` | Boolean mask for validation nodes. |
| `test_mask` | `N x 1` | Boolean mask for final test nodes. |

### 2. Data Loading

`load_data.m` supports three dataset modes:

- `toy`: creates a small synthetic hypergraph with class-centered features. This is useful for quick sanity checks because it runs almost immediately.
- `cora`: reads `cora.content` and `cora.cites`, converts paper labels to one-hot vectors, normalizes feature rows, builds research-group hyperedges, and creates train/validation/test masks.
- `custom`: reserved for user-defined datasets. This branch intentionally raises an error until a custom loader is implemented.

For Cora, `read_cora_content` parses each paper ID, feature vector, and class label from `cora.content`. Features are stored as a sparse matrix and row-normalized so each paper feature vector has comparable scale.

### 3. Hypergraph Construction

The Cora citation file describes pairwise citation relations, but HGNN needs hyperedges that can connect more than two nodes. This implementation turns citations into research-group hyperedges:

1. Count how often each paper is cited.
2. Select papers with citation count greater than or equal to `min_group_citations`.
3. For each selected seed paper, create one hyperedge containing the seed paper and every paper that cites it.
4. Optionally add singleton hyperedges for papers that do not appear in any research-group hyperedge.

This converts local citation neighborhoods into higher-order groups. A single propagation step can then move information across all papers in the same research context, not just across one citation edge at a time.

### 4. Propagation Matrix

`compute_laplacian.m` converts the incidence matrix `H` into the normalized HGNN propagation matrix:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^\top \cdot D_v^{-1/2}$$

The intermediate matrices mean:

- `W`: diagonal hyperedge-weight matrix. If no weights are provided, all hyperedges receive weight `1`.
- `D_v`: diagonal node-degree matrix. A node degree is the weighted number of hyperedges that contain the node.
- `D_e`: diagonal hyperedge-degree matrix. A hyperedge degree is the number of nodes inside that hyperedge.
- `Theta_conv`: normalized propagation matrix used by every HGNN layer.

The normalization prevents high-degree nodes or large hyperedges from dominating feature propagation.

### 5. Forward Pass

The model is a two-layer HGNN:

```matlab
H1     = ReLU(Theta_conv * X  * W1)
Y_pred = Softmax(Theta_conv * H1 * W2)
```

`hgnn_layer.m` handles one layer by applying hypergraph propagation, a learnable linear transform, and an activation. `hgnn_forward.m` chains two layers and returns both predictions and cached intermediate values for backpropagation.

### 6. Training Loop

`train.m` performs manual training without a deep learning toolbox:

1. Initialize `W1` and `W2` with Xavier initialization.
2. Run the forward pass.
3. Compute masked cross-entropy loss on `train_mask` nodes only.
4. Run `hgnn_backward.m` to compute gradients.
5. Add L2 weight-decay gradients.
6. Update weights with Adam.
7. Print train and validation accuracy every `print_every` epochs.

Validation nodes are never used to compute the training loss. They are only used for progress reporting during training.

### 7. Final Evaluation

After training, `main.m` runs one final forward pass and calls `evaluate.m` with `test_mask`. The reported test accuracy is therefore computed only on the held-out test nodes.

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

Cora hyperedges are built with a **research-group construction centered on frequently cited papers**:

```
Paper A is cited at least 5 times.
Papers that cite Paper A: {B, C, D, E, ...}
   -> Research-group hyperedge = {A, B, C, D, E, ...}
```

By default, a paper becomes a research-group seed when it has at least `min_group_citations = 5` incoming citations. The seed and the papers that cite it are grouped into one hyperedge, representing a higher-order relationship among papers that share a research direction.

The threshold can be adjusted in `main.m`:

```matlab
data_options.min_group_citations = 5;
data_options.include_singletons  = true;
```

Useful behavior changes:

- Lower `min_group_citations` to create more research-group hyperedges.
- Raise `min_group_citations` to keep only stronger citation groups.
- Set `max_research_groups` to limit how many high-citation seeds are used.
- Set `include_singletons = false` to skip isolated papers that do not belong to any research-group hyperedge.

> **Note:** When `include_singletons = true`, nodes that do not belong to any research-group hyperedge are automatically added as singleton hyperedges.


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
