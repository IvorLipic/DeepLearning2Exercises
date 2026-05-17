# Duboko učenje 2 — Laboratorijske vježbe (Generativni modeli)

---

## Contents

| Lab | Topic | Framework | Description |
|-----|-------|-----------|-------------|
| 1 | Uvod u generativne modele | TensorFlow | Gaussian & Uniform Mixture Models; Probabilistic Regression |
| 2 | Varijacijski autoenkoder (VAE) | TensorFlow + SymPy | Monte Carlo estimation, importance sampling, reparameterization trick, VAE |
| 3 | Boltzmannovi strojevi | PyTorch | RBM (CD-k), Deep Belief Networks (DBN), MNIST |
| 4 | GAN / DCGAN | PyTorch | Deep Convolutional GAN for MNIST; training dynamics analysis |
| 5 | Normalizirajući tokovi | PyTorch (normflows) | Residual Flows, Real NVP; `make_moons` dataset |

---

## Lab 1 — Uvod u generativne modele

**Files:** `lab1/dists.py`, `lab1/models.py`, `lab1/graphics.py`, `lab1/lab_1.ipynb`

### Section 1 — Mixture Models

- **Gaussian Mixture Model (`GMDist`):** K components, each with `π_k`, `μ_k`, `σ²_k`. Sampling via categorical choice + normal draw. PDF as weighted sum of Gaussians.
- **Uniform Mixture Model (`UMDist`):** Same structure with uniform components.
- **Maximum Likelihood Training:** Parameters reparameterized as `logπᵢ` → softmax, `logσ²` → exp to avoid constraints. Loss = negative log-likelihood using `LogSumExp` for numerical stability. Optimized with Adam (lr=1e-2, 5000 epochs).
- **Exercises:**
  - a) Generate random GMM (K=3-5), sample L=1e6 points, plot histogram + PDF
  - b) Implement `GMModel.loss()`, `p_xz()`, `p_x()`
  - c) Compare learned vs true density
  - d) Compare learned vs true component weights
  - e) Clustering via posterior `p(zₖ|x)` — assign each point to argmax component
  - f-g) Fit GMM to uniform mixture data; vary K (3-10) to see approximation limits
  - h (implicit): Analyze whether GMM can approximate non-Gaussian distributions

### Section 2 — Probabilistic Regression

- Model `y|x ~ N(μ(x), σ²(x))` where both μ and σ² are outputs of a neural network.
- Loss = negative log-likelihood (heteroscedastic).
- Visualize 1-σ confidence intervals alongside scatter plot.

---

## Lab 2 — Varijacijski autoenkoder (VAE)

**Files:** `lab2/dists.py`, `lab2/models.py`, `lab2/utils.py`, `lab2/tf_utils.py`, `lab2/printing.py`, `lab2/graphics.py`, `lab2/lab_2.ipynb`

### Section 1 — Monte Carlo Estimation

- Define `p(z) = ¾(1 − z²)` on [-1, 1]. Generate random process `f(z, t)` symbolically via `gen_f_zt()`.
- Analytical expectation `g(t) = ∫ f(z,t) p(z) dz` computed with SymPy.
- MC estimate `ĝᴸ(t) = ¹⁄ᴸ Σ f(z^(l), t)` for L ∈ {1, 10, 100, 1000}. Larger L reduces variance.

### Section 2 — Importance Sampling

- Optimal proposal `q(z) ∝ f(z,t) p(z)` computed analytically (via `gen_qz()`).
- Inverse CDF `Q⁻¹(ε)` approximated via binary search (since Q⁻¹ has no closed form).
- Compare MC vs ISMC estimators: ISMC has lower variance, tighter confidence intervals.

### Section 3 — Reparameterization Trick

- Express `z = g(ε, φ)` where `ε ~ p(ε)` is parameter-free (e.g., `z = μ + σ·ε`).
- Allows gradients to flow through the sampling step.
- Compared to score-function estimator: reparameterization gives lower-variance gradients.

### Section 4 — Variational Inference

- ELBO derivation: `log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))`.
- KL divergence between two Gaussians computed in closed form.

### Section 5 — VAE Training

- Encoder: outputs `μ` and `logσ²` for `q(z|x)`.
- Decoder: reconstructs `x` from `z`.
- Trained by maximizing ELBO. Compare unconditional generation (sample z ~ N(0,I)) vs conditional reconstruction.

---

## Lab 3 — Boltzmannovi strojevi

**Files:** `lab3/Duboko_ucenje_laboratorijska_vjezba_generativni_modeli_Boltzmannovi_strojevi.ipynb`

### Task 1 — Restricted Boltzmann Machine (RBM)

- **Architecture:** Visible = 784 (MNIST pixels), Hidden = 100. Weights initialized N(0, 0.1).
- **Training:** Contrastive Divergence (CD-k). Parameters updated via:
  - `ΔW = η (⟨vᵢhⱼ⟩⁰ − ⟨vᵢhⱼ⟩ᵏ)`
  - `Δb = η (⟨hⱼ⟩⁰ − ⟨hⱼ⟩ᵏ)`, `Δa = η (⟨vᵢ⟩⁰ − ⟨vᵢ⟩ᵏ)`
- **Experiments:**
  - Train for 100 epochs, batch size 100, learning rate 0.1
  - **Sub-task 1:** Visualize learned weight filters (10×10 grid of 28×28 patches)
  - **Sub-task 2:** Reconstruct first 10 test MNIST digits → compare original, reconstruction, hidden state
  - **Sub-task 3:** Random hidden initialization → generate via Gibbs sampling
  - **Sub-task 4:** Save weights to `zad1_rbm.th`
- **Question:** Effect of increasing CD-k steps → more accurate gradient but slower training; CD-1 biased toward reconstructions, higher k yields better generative samples

### Task 2 — Deep Belief Network (DBN)

- Stacked RBMs trained greedily layer-by-layer.
- Fine-tune with a classifier head.
- Save to `zad2_dbn.th` and `zad3_dbn_ft.th`.

---

## Lab 4 — Generativne suparničke mreže (DCGAN)

**File:** `lab4/Duboko učenje - laboratorijska vježba - generativni modeli - Generative adversarial networks (GAN).ipynb`

### Generator Architecture

| Layer | Input → Output | Kernel | Stride | Padding | BN |
|-------|---------------|--------|--------|---------|----|
| 1 | z(100,1,1) → 512×4×4 | 4 | 1 | 0 | Yes |
| 2 | 512×4×4 → 256×8×8 | 4 | 2 | 1 | Yes |
| 3 | 256×8×8 → 128×16×16 | 4 | 2 | 1 | Yes |
| 4 | 128×16×16 → 64×32×32 | 4 | 2 | 1 | Yes |
| 5 | 64×32×32 → 1×64×64 | 4 | 2 | 1 | No |

Activations: LeakyReLU(0.2) on hidden layers, Tanh on output.

### Discriminator Architecture

| Layer | Input → Output | Kernel | Stride | Padding | BN |
|-------|---------------|--------|--------|---------|-----|
| 1 | 1×64×64 → 64×32×32 | 4 | 2 | 1 | No |
| 2 | 64×32×32 → 128×16×16 | 4 | 2 | 1 | Yes |
| 3 | 128×16×16 → 256×8×8 | 4 | 2 | 1 | Yes |
| 4 | 256×8×8 → 512×4×4 | 4 | 2 | 1 | Yes |
| 5 | 512×4×4 → 1 (scalar) | 4 | 1 | 0 | No |

Activations: LeakyReLU(0.2) on hidden layers, Sigmoid on output. Optimizer: Adam(β₁=0.5).

### Experiments & Findings

| Condition | Result |
|-----------|--------|
| **D:1x, G:1x (baseline)** | Oscillating D/G loss; recognizable digits |
| **D:2x, G:1x** | D dominates → mode collapse (d_loss → 200, g_loss → 0 by epoch 12) |
| **D:1x, G:2x** | G keeps up → diverse outputs (7s, 3s, 0s), then collapses around epoch 11 |
| **No BatchNorm** | Sharp images initially, but numerical explosion by epoch 14; training diverges |

---

## Lab 5 — Normalizirajući tokovi

**File:** `lab5/lab_5_NF.ipynb`

### Residual Flows

- **Architecture:** `f_l(x) = x + g_l(x)` with Lipschitz(g_l) ≤ κ = 0.9.
- Invertible residual block using `LipschitzMLP`. Uses `ActNorm` between blocks.
- Jacobian not explicitly triangular — uses unbiased log-det estimator (reduced memory).
- **Parameters:** 16 layers, hidden_units=128, hidden_layers=3 per block.

### Real NVP

- **Affine coupling:** Split input in two halves; one half is scaled/shifted based on the other.
  - Forward: `y₁:d = x₁:d`, `y_(d+1:D) = x_(d+1:D) ⊙ exp(s(x₁:d)) + t(x₁:d)`
  - Reverse: `x₁:d = y₁:d`, `x_(d+1:D) = (y_(d+1:D) − t(y₁:d)) ⊙ exp(−s(y₁:d))`
- Jacobian is triangular → log-det = sum of scale parameters.
- **Parameters:** `MLP([1, hidden_units, hidden_units, 2])` for s and t networks; `Permute` layers alternate which half is transformed.

### Experiments

- **Dataset:** `make_moons(2²⁰)` with noise ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}.
- **Training:** 5000 epochs, `forward_kld()` loss, Adam optimizer. Residual flows require `update_lipschitz()` every iteration.
- **Noise effect:**
  - Low noise (0.05): Sharp density, concentrated along moon crescents → lower loss, overfitting risk
  - High noise (0.3): Wider/smoother distribution, moons blend → higher loss (more entropy)
- **Model comparison:**
  - NVP trains faster (~40 it/s for L=16, U=128), good coverage
  - Residual slower (~4 it/s) but more expressive

---

## Requirements

- Python 3.8+
- TensorFlow 2.x (labs 1-2)
- PyTorch 1.x + torchvision (labs 3-5)
- normflows (lab 5)
- numpy, matplotlib, scipy, sympy, seaborn, tqdm, pandas
- Jupyter Notebook / JupyterLab

## Acknowledgments

University of Zagreb, Faculty of Electrical Engineering and Computing (FER)  
Course: [Duboko učenje 2](https://www.fer.unizg.hr/predmet/dubuce2)
