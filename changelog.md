# HiRISE Anomaly Detection — Model Evolution Changelog

**Project:** HiRISE Anomaly Detection — Unsupervised Deep Generative Modeling on Martian Surface Imagery  
**Competition:** National Space Science Conclave (NSSC) 2026 — Data Analytics Case Competition, Round 1  
**Team:** ASTRA — Astronomy Club, Indian Institute of Technology (BHU) Varanasi  
**Purpose of this file:** Document the complete model-development journey from Version 1 to Version 3 using the actual submitted notebooks, recorded outputs, and final report.

---

## 1. Changelog Objective

The competition problem requires an unsupervised generative approach that learns the distribution of dominant/normal HiRISE Martian imagery and uses that learned representation to identify structurally abnormal frames **without anomaly labels during model training**.

The required evolution format is:

> **Symptom → Diagnosis → Fix**

This changelog therefore does not merely list models. Each version records:

1. **What was built**
2. **What was observed**
3. **What problem the observation revealed**
4. **Why the next modeling change was selected**
5. **What changed technically**
6. **What the next experiment achieved**
7. **What was learned from the experiment**

The progression was:

```text
Version 1
Convolutional Autoencoder (CAE)
        │
        │  reconstruction baseline
        ▼
Version 2
Convolutional Variational Autoencoder (CVAE)
        │
        │  probabilistic latent regularization
        ▼
Version 3
f-AnoGAN
WGAN-GP + dedicated Encoder
        │
        │  reconstruction + discriminator feature discrepancy
        ▼
Final selected model
f-AnoGAN
```

The problem statement explicitly requires a Markdown changelog covering at least three versions using the **Symptom → Diagnosis → Fix** structure.

---

# 2. Source Notebooks and Experiment Mapping

| Version | Model | Submitted notebook | Main purpose |
|---|---|---|---|
| **V1** | Convolutional Autoencoder (CAE) | `v1-auto-encoder.ipynb` | Establish a deterministic reconstruction-based baseline |
| **V2** | Convolutional Variational Autoencoder (CVAE) | `v2-variational-autencoderp.ipynb` | Test whether probabilistic latent regularization improves anomaly separation |
| **V3** | f-AnoGAN | `v3-f-AnoGAN.ipynb` | Learn a normal image manifold adversarially and combine reconstruction with discriminator feature discrepancy |

The final report is based on these experiments and records the measured progression:

| Version | Model | AUROC | Threshold |
|---|---|---:|---:|
| V1 | CAE | **0.9163** | 0.00823 |
| V2 | CVAE | **0.8599** | 0.04929 |
| V3 | f-AnoGAN | **0.9687** | 0.100311 |

**Important:** the models do not use identical preprocessing or anomaly-score definitions. Therefore the AUROC ranking is the measured empirical result, while architectural comparisons must also consider input resolution and scoring methodology.

---

# 3. Dataset and Problem Setup

## 3.1 Dataset structure

The notebooks use the following structure:

```text
test+train/
├── train/
│   └── normal/
└── test/
    ├── normal/
    └── anomaly_real/
```

The intended learning setup is:

- `train/normal` → model fitting
- `test/normal` → normal reference/evaluation
- `test/anomaly_real` → real anomaly evaluation

The competition problem identifies three broad anomaly categories:

1. **Sensor Artifact**
   - striping
   - saturation
   - diagonal cut-offs
   - acquisition-related sensor defects

2. **Corrupted Crop**
   - missing or truncated content
   - malformed crops
   - black borders
   - tiling/seam artifacts

3. **Non-Martian Content**
   - imagery that does not correspond to genuine Martian terrain
   - examples include calibration/spacecraft-related imagery

The central constraint is that anomaly labels are **not used for fitting the generative models**.

---

## 3.2 Dataset counts observed across notebooks

### Version 1

The V1 DataLoader contains:

- **314 batches**
- **batch size = 64**

The explicit scoring cell reports:

- **2,100 normal test images**
- **5,125 anomaly test images**

The train directory contains only:

```text
normal
```

so the model is trained on normal imagery.

### Version 2

The executed notebook reports:

- **20,000 training images**
- **2,100 normal test images**
- **5,125 anomaly test images**

### Version 3

The executed f-AnoGAN notebook reports:

- **20,000 normal training images**
- **2,100 normal test images**
- **5,125 anomaly test images**
- **7,225 total test images**

The small difference between the V1/V2 and V3 normal counts is retained as an implementation detail rather than silently corrected.

---

# 4. Version 1 — Convolutional Autoencoder

## 4.1 Experiment identity

**Notebook:** `v1-auto-encoder.ipynb`  
**Model:** Convolutional Autoencoder (CAE)  
**Role:** First reconstruction-based baseline

The purpose of V1 was to answer the most basic question:

> Can a model trained only on normal Martian imagery reconstruct normal images better than abnormal images?

The model learns:

```text
x → Encoder → latent representation → Decoder → reconstruction x̂
```

and uses reconstruction error as the anomaly signal.

---

# 5. V1 — Symptom → Diagnosis → Fix

## 5.1 Entry V1-01: Establish the baseline

### Symptom

Before introducing probabilistic or adversarial modeling, there was no empirical baseline showing how well a simple convolutional reconstruction model could separate normal and anomalous HiRISE images.

### Diagnosis

A deterministic convolutional autoencoder is a natural first experiment because it:

- can be trained using only normal images;
- learns a compact representation of normal imagery;
- produces a reconstruction;
- naturally provides an anomaly score through reconstruction error.

### Fix

Implement a convolutional autoencoder and train it only on the normal training directory.

### Result

The baseline achieved:

- **AUROC = 0.9163**

This established that reconstruction error already contains substantial information for separating normal and anomalous images.

---

# 6. V1 Architecture

The submitted V1 notebook uses:

```text
Input: 3 × 224 × 224

Encoder:
    Conv2d 3  → 16
    MaxPool
    Conv2d 16 → 32
    MaxPool
    Conv2d 32 → 64
    MaxPool

Bottleneck:
    64 × 28 × 28

Decoder:
    ConvTranspose2d 64 → 32
    ConvTranspose2d 32 → 16
    ConvTranspose2d 16 → 3

Output:
    3 × 224 × 224
```

Activations:

- ReLU in intermediate layers
- Sigmoid at the final decoder layer

Spatial progression:

```text
224 × 224
    ↓
112 × 112
    ↓
56 × 56
    ↓
28 × 28
    ↓
56 × 56
    ↓
112 × 112
    ↓
224 × 224
```

---

# 7. V1 Training Configuration

| Parameter | Value |
|---|---|
| Input | 224 × 224 RGB |
| Batch size | 64 |
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Epochs | 10 |
| Recorded device | CUDA |
| Scheduler | None |
| Explicit early stopping | None |

Preprocessing:

```text
Resize(224, 224)
        ↓
ToTensor()
        ↓
pixel range [0, 1]
```

No [-1, 1] normalization was used in V1.

---

# 8. V1 Training Behaviour

The notebook prints the final batch loss at every epoch. These are **final-batch losses, not epoch-average losses**.

| Epoch | Final batch MSE |
|---:|---:|
| 1 | 0.0018 |
| 2 | 0.0011 |
| 3 | 0.0014 |
| 4 | 0.0006 |
| 5 | 0.0006 |
| 6 | 0.0005 |
| 7 | 0.0009 |
| 8 | 0.0007 |
| 9 | 0.0010 |
| 10 | 0.0010 |

### Observation

The reconstruction loss falls rapidly from the first epoch and remains around the 10^-3 range or lower.

The fluctuations after the initial reduction are not interpreted as instability because the notebook records only the **last batch of each epoch**, rather than an epoch mean.

---

# 9. V1 Anomaly Scoring

An important implementation detail is that V1 does **not** use a simple global mean image reconstruction error.

For an input image `x` and reconstruction `x̂`:

```text
e = (x̂ - x)^2
```

The error is average-pooled into local patches.

The submitted scoring code uses:

```text
patch = IMG_SIZE / 8
      = 224 / 8
      = 28
```

so the 224 × 224 error image is summarized into an 8 × 8 local-error grid.

The final image score is:

```text
s(x) = max over spatial locations and channels of pooled error
```

### Why this matters

This makes V1 particularly sensitive to a **localized high-error region** rather than only a globally elevated reconstruction error.

This is a deliberate anomaly-detection design rather than an accidental side effect.

---

# 10. V1 Thresholding

The submitted notebook selects:

```text
threshold = 95th percentile of NORMAL TEST scores
```

Measured threshold:

```text
τ = 0.00823
```

Decision rule:

```text
if score > 0.00823:
    anomaly
else:
    normal
```

### Important methodological note

This threshold is **not** a training-only calibration threshold.

The V1 notebook explicitly computes the threshold from the normal test-score distribution.

Therefore the changelog and report do not describe this threshold as a clean held-out-training calibration procedure.

---

# 11. V1 Results

Measured results:

| Metric | V1 CAE |
|---|---:|
| Normal test images | 2,098 |
| Anomaly test images | 5,125 |
| Mean normal score | 0.00300 |
| Mean anomaly score | 0.00772 |
| AUROC | **0.9163** |
| Threshold | **0.00823** |
| Normal flagged rate | 0.050 |
| Anomaly flagged rate | 0.334 |
| Precision | 0.9421 |
| Recall | 0.3337 |
| F1 | 0.4928 |
| Accuracy | 0.5127 |

Confusion matrix:

| Actual / Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | 1,993 | 105 |
| Anomaly | 3,415 | 1,710 |

---

# 12. V1 — Key Symptom

The most important V1 observation was:

```text
AUROC = 0.9163
```

but at the selected threshold:

```text
Recall = 0.3337
```

and:

```text
Anomaly flagged rate = 0.334
```

### Symptom

The model has good overall ranking ability, but the chosen threshold produces a conservative operating point and misses a large fraction of anomalous images.

### Diagnosis

Two different things are being measured:

1. **AUROC** asks whether anomaly scores rank anomalous images above normal images across all thresholds.
2. **Thresholded recall** asks how many anomalies are actually flagged at one specific threshold.

The V1 result demonstrates that strong ranking quality does not automatically produce a high-recall operating point.

### Fix / Evolution

The next version was not designed simply to change the threshold. Instead, the team tested whether a **better latent representation** could improve the underlying anomaly score itself.

This motivated Version 2.

---

# 13. Version 1 → Version 2

## Transition

```text
V1 CAE
   ↓
V2 CVAE
```

### Symptom

The deterministic autoencoder provided strong reconstruction-based separation, but its latent representation was not explicitly regularized to follow a known probability distribution.

### Diagnosis

The CAE learns a deterministic latent representation:

```text
z = E(x)
```

There is no KL regularization or explicit probabilistic prior on the latent space.

This motivated testing whether a smoother, probabilistically regularized latent representation could improve generalization and anomaly separation.

### Fix

Replace the deterministic bottleneck with a variational bottleneck that predicts:

```text
μ(x)
log σ²(x)
```

and samples:

```text
z = μ + σ ⊙ ε
ε ~ N(0, I)
```

The VAE objective becomes:

```text
L = Lreconstruction + β LKL
```

with:

```text
LKL = DKL(q(z|x) || N(0,I))
```

In the actual V2 implementation, `β = 0.5`.

---

# 14. Version 2 — Convolutional Variational Autoencoder

## 14.1 Experiment identity

**Notebook:** `v2-variational-autencoderp.ipynb`  
**Model:** Convolutional VAE (CVAE)

V2 was a direct architectural evolution of V1.

Instead of:

```text
image → deterministic latent → reconstruction
```

V2 uses:

```text
image
  ↓
convolutional encoder
  ↓
μ, log variance
  ↓
reparameterization
  ↓
latent z
  ↓
decoder
  ↓
reconstruction
```

---

# 15. V2 Data and Preprocessing

The executed V2 notebook reports:

```text
Train: 20000 images
Test normal: 210 images
Test anomaly: 5125 images
```

The model converts each image to grayscale:

```text
RGB
 ↓
grayscale
 ↓
128 × 128
 ↓
[0, 1]
```

Configuration:

| Parameter | Value |
|---|---|
| Image size | 128 × 128 |
| Channels | 1 |
| Batch size | 128 |
| Latent dimension | 64 |
| Base channels | 32 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 30 |
| KL weight β | 0.5 |
| Device | CUDA |
| Random seeds | PyTorch 0, NumPy 0 |

---

# 16. V2 Architecture

Encoder:

```text
1 channel
   ↓
32 channels
   ↓
64 channels
   ↓
128 channels
   ↓
256 channels
```

Each encoder block uses:

```text
Conv2d
ReLU
```

with stride 2.

At 128 × 128 input resolution, the spatial resolution becomes:

```text
128 → 64 → 32 → 16 → 8
```

The encoded feature representation is flattened.

Two separate linear layers produce:

```text
μ
logvar
```

for a latent dimension of 64.

The decoder maps the latent vector back through:

```text
Linear
 ↓
8 × 8 feature representation
 ↓
ConvTranspose
 ↓
16 × 16
 ↓
32 × 32
 ↓
64 × 64
 ↓
128 × 128
```

with a final Sigmoid output.

---

# 17. V2 Loss Function

The implemented V2 loss is:

```text
Ltotal = Lreconstruction + 0.5 × LKL
```

where:

```text
Lreconstruction = MSE(reconstruction, input)
```

and:

```text
LKL =
-0.5 × Σ(1 + logvar - μ² - exp(logvar))
```

normalized per batch.

### Why this was introduced

The reconstruction term still teaches the model to reproduce normal images.

The KL term additionally encourages the learned latent distribution to remain close to a unit Gaussian.

The experiment therefore tested whether this additional structure would improve anomaly detection.

---

# 18. V2 Training Results

The executed V2 notebook ran for **30 epochs**.

| Epoch | Train reconstruction | KL divergence | Validation reconstruction |
|---:|---:|---:|---:|
| 1 | 186.6005 | 15.7972 | 82.9358 |
| 2 | 74.5720 | 17.9489 | 70.1559 |
| 3 | 67.6637 | 16.2957 | 64.7310 |
| 4 | 62.6449 | 17.5150 | 60.7700 |
| 5 | 60.4286 | 18.9347 | 59.3561 |
| 6 | 59.0781 | 19.8492 | 58.6441 |
| 7 | 58.0274 | 20.2136 | 57.6370 |
| 8 | 57.3235 | 19.9856 | 56.8429 |
| 9 | 56.2180 | 19.5976 | 55.7661 |
| 10 | 55.4945 | 19.9458 | 55.1127 |
| 11 | 55.4672 | 20.1849 | 54.8863 |
| 12 | 54.5693 | 20.4182 | 54.4337 |
| 13 | 53.9387 | 20.5618 | 53.6841 |
| 14 | 53.5215 | 20.7196 | 53.3346 |
| 15 | 53.3026 | 20.7623 | 53.1384 |
| 16 | 53.0042 | 20.7996 | 53.3974 |
| 17 | 52.6768 | 20.6473 | 54.0462 |
| 18 | 52.3734 | 20.5828 | 52.9839 |
| 19 | 51.9636 | 20.4704 | 52.2493 |
| 20 | 51.8550 | 20.2756 | 52.1776 |
| 21 | 51.8283 | 20.0586 | 51.8321 |
| 22 | 52.0493 | 19.9161 | 52.3809 |
| 23 | 51.5438 | 19.9350 | 52.3284 |
| 24 | 51.3813 | 19.9331 | 51.3975 |
| 25 | 51.3672 | 19.9697 | 51.8978 |
| 26 | 51.0616 | 20.0606 | 51.7256 |
| 27 | 50.9688 | 20.0667 | 51.7803 |
| 28 | 50.7794 | 20.1149 | 51.3771 |
| 29 | 50.8396 | 20.2005 | 50.9153 |
| 30 | 50.6008 | 20.2092 | 51.4464 |

The training run took approximately:

```text
642.35 seconds
```

The lowest recorded validation reconstruction loss was:

```text
50.9153 at epoch 29
```

---

# 19. V2 Training Interpretation

From epoch 1 to epoch 30:

```text
Train reconstruction:
186.6005 → 50.6008

KL:
15.7972 → 20.2092

Validation reconstruction:
82.9358 → 51.4464
```

### Symptom

The reconstruction objective improved substantially.

However, the anomaly-detection metric did not improve relative to V1.

### Diagnosis

The CVAE successfully optimized its training objective, but lower reconstruction loss alone did not translate into better anomaly ranking.

This is an important distinction:

> Better optimization of the generative objective does not necessarily mean better anomaly discrimination.

The probabilistic latent regularization therefore did not provide an empirical improvement on the current test set.

### Fix / Evolution

Rather than continuing to make incremental changes inside the same reconstruction/VAE framework, the next experiment changed the type of signal used for anomaly detection.

This motivated f-AnoGAN.

---

# 20. V2 Anomaly Scoring

V2 uses the same style of local reconstruction scoring as V1:

```text
error = (reconstruction - input)^2
```

then:

```text
average pooling
```

followed by:

```text
maximum pooled error
```

The image is 128 × 128 and the patch size is:

```text
128 / 8 = 16
```

so the error is summarized into an 8 × 8 grid.

---

# 21. V2 Threshold and Results

Measured threshold:

```text
τ = 0.04929
```

The threshold is the 95th percentile of normal test scores.

Measured mean scores:

```text
Normal:  0.01908
Anomaly: 0.04917
```

Measured AUROC:

```text
AUROC = 0.8599
```

Normal flagged rate:

```text
0.050
```

Anomaly flagged rate:

```text
0.407
```

### Comparison against V1

```text
V1 CAE   AUROC = 0.9163
V2 CVAE  AUROC = 0.8599
```

Therefore:

```text
0.8599 < 0.9163
```

The CVAE did **not** empirically improve the anomaly ranking over the CAE.

---

# 22. V2 → V3 — Major Model Evolution

## Symptom

The CVAE produced a lower AUROC than the simpler CAE:

```text
CAE   = 0.9163
CVAE  = 0.8599
```

The reconstruction-based approach was therefore not benefiting from the additional probabilistic latent structure on this dataset.

## Diagnosis

The measured evidence supports the following conclusion:

- the VAE objective was optimizing reconstruction and KL regularization;
- training reconstruction error decreased substantially;
- validation reconstruction error also decreased;
- nevertheless, anomaly ranking was weaker than the CAE.

A further architectural change was therefore justified.

A **hypothesis**, rather than a directly proven diagnosis, was that a purely reconstruction-driven anomaly signal may not sufficiently represent all structural differences between an input and the normal image manifold.

Instead of claiming that the CVAE definitively suffered from one specific pathology such as posterior collapse or blur, the experiment moved to a model with a different anomaly representation.

## Fix

Introduce **f-AnoGAN**:

```text
WGAN-GP
   +
dedicated encoder
   +
pixel reconstruction discrepancy
   +
discriminator feature discrepancy
```

The goal was to learn a stronger normal-image manifold and obtain an anomaly score containing both:

1. pixel-level disagreement;
2. learned feature-space disagreement.

---

# 23. Version 3 — f-AnoGAN

## 23.1 Experiment identity

**Notebook:** `v3-f-AnoGAN.ipynb`  
**Model:** f-AnoGAN  
**Architecture:** WGAN-GP + Encoder

V3 is a substantially different architecture.

Instead of learning only:

```text
image → reconstruction
```

the system first learns a generative model of normal imagery:

```text
latent z → Generator → normal-looking image
```

and then learns an encoder:

```text
image → Encoder → z → Generator → reconstruction
```

The critic additionally provides a feature representation used for anomaly scoring.

---

# 24. V3 Data Pipeline

The executed notebook extracts the dataset and reports:

```text
Training images:      20,000
Test normal images:    2,100
Test anomaly images:   5,125
```

Input:

```text
64 × 64 RGB
```

Batch size:

```text
32
```

Normalization:

```text
x_normalized = (x - 0.5) / 0.5
```

so the target range is approximately:

```text
[-1, 1]
```

This matches the generator's final:

```text
Tanh
```

activation.

---

# 25. V3 Configuration

| Parameter | Value |
|---|---:|
| Image size | 64 × 64 |
| Channels | 3 |
| Batch size | 32 |
| Latent dimension | 128 |
| Base channels | 64 |
| DataLoader workers | 0 |
| GAN optimizer | Adam |
| GAN learning rate | 2e-4 |
| GAN betas | (0.0, 0.9) |
| Critic updates / generator update | 5 |
| Gradient penalty weight | 10 |
| Maximum GAN epochs | 70 |
| Early-stopping patience | 15 |
| Minimum improvement | 0.001 |
| Encoder epochs | 50 |
| Encoder learning rate | 2e-4 |
| Encoder optimizer | Adam |
| Encoder betas | (0.5, 0.999) |
| Reconstruction loss | L1 |
| Feature loss | L1 |
| Reconstruction weight | 0.9 |
| Feature weight | 0.1 |

---

# 26. V3 Generator

The generator starts from:

```text
z ∈ R^128
```

and progressively upsamples:

```text
128 × 1 × 1
      ↓
512 × 4 × 4
      ↓
256 × 8 × 8
      ↓
128 × 16 × 16
      ↓
64 × 32 × 32
      ↓
3 × 64 × 64
```

Layers use:

- ConvTranspose2d
- BatchNorm
- ReLU

with a final:

```text
Tanh
```

output.

---

# 27. V3 Critic

The critic is a WGAN critic, so it does **not** use a final Sigmoid.

Its feature extractor progresses:

```text
3 × 64 × 64
      ↓
64 × 32 × 32
      ↓
128 × 16 × 16
      ↓
256 × 8 × 8
      ↓
512 × 4 × 4
```

The final critic layer maps:

```text
512 × 4 × 4 → scalar
```

The intermediate:

```text
512 × 4 × 4
```

feature map is retained.

This feature representation is important because f-AnoGAN later compares the critic features of:

```text
real image
```

and:

```text
reconstructed image
```

---

# 28. V3 Encoder

After GAN training, a separate encoder is trained.

The encoder mirrors the critic's spatial reduction:

```text
3 × 64 × 64
      ↓
64 × 32 × 32
      ↓
128 × 16 × 16
      ↓
256 × 8 × 8
      ↓
512 × 4 × 4
      ↓
128 × 1 × 1
      ↓
128-dimensional latent vector
```

This allows inference to be feed-forward:

```text
x → E(x) → z → G(z)
```

instead of requiring iterative latent optimization for each test image.

---

# 29. V3 Weight Initialization

The submitted notebook applies DCGAN-style initialization:

### Convolution weights

```text
N(0, 0.02)
```

### BatchNorm weights

```text
N(1, 0.02)
```

### BatchNorm biases

```text
0
```

This initialization is applied to:

- generator
- critic
- encoder

---

# 30. V3 Stage 1 — WGAN-GP

The critic loss implemented is:

```text
LD =
E[D(fake)]
-
E[D(real)]
+
10 × gradient_penalty
```

The generator loss is:

```text
LG = -E[D(fake)]
```

The gradient penalty is calculated from random interpolations between real and generated images.

The objective is to enforce a smoother critic and improve adversarial training stability.

---

# 31. V3 GAN Training — Epoch-by-Epoch Record

The executed notebook completed all 70 epochs.

| Epoch | Critic loss | Generator loss | Status |
|---:|---:|---:|---|
| 1 | -4.4840 | 30.6199 | Improved |
| 2 | -4.4708 | 31.9287 | No improvement 1/15 |
| 3 | -4.5327 | 31.9427 | No improvement 2/15 |
| 4 | -4.5089 | 32.8813 | No improvement 3/15 |
| 5 | -4.2875 | 32.6063 | No improvement 4/15 |
| 6 | -4.4674 | 33.3640 | No improvement 5/15 |
| 7 | -4.4594 | 33.2203 | No improvement 6/15 |
| 8 | -4.1192 | 33.0699 | No improvement 7/15 |
| 9 | -3.9842 | 32.3873 | No improvement 8/15 |
| 10 | -3.8882 | 32.3978 | No improvement 9/15 |
| 11 | -3.8345 | 30.8113 | No improvement 10/15 |
| 12 | -3.5523 | 29.8675 | Improved |
| 13 | -3.2557 | 27.5509 | Improved |
| 14 | -3.4679 | 26.9196 | Improved |
| 15 | -3.4456 | 24.8014 | Improved |
| 16 | -3.2281 | 24.6541 | Improved |
| 17 | -2.9060 | 23.1197 | Improved |
| 18 | -2.9961 | 23.2905 | No improvement 1/15 |
| 19 | -2.7771 | 22.2458 | Improved |
| 20 | -2.5541 | 21.3522 | Improved |
| 21 | -2.4464 | 20.1348 | Improved |
| 22 | -2.0036 | 17.1470 | Improved |
| 23 | -2.0272 | 17.3481 | No improvement 1/15 |
| 24 | -2.0763 | 17.5258 | No improvement 2/15 |
| 25 | -2.0003 | 16.8789 | Improved |
| 26 | -1.9306 | 16.2321 | Improved |
| 27 | -1.8171 | 14.6042 | Improved |
| 28 | -1.8786 | 14.3014 | Improved |
| 29 | -1.9059 | 14.3385 | No improvement 1/15 |
| 30 | -1.8389 | 14.2200 | Improved |
| 31 | -1.8245 | 13.9488 | Improved |
| 32 | -1.8182 | 13.8840 | Improved |
| 33 | -1.7734 | 13.4723 | Improved |
| 34 | -1.6794 | 12.9368 | Improved |
| 35 | -1.6915 | 12.4718 | Improved |
| 36 | -1.6556 | 11.8992 | Improved |
| 37 | -1.6379 | 10.9074 | Improved |
| 38 | -1.6461 | 10.9250 | No improvement 1/15 |
| 39 | -1.5238 | 10.0246 | Improved |
| 40 | -1.5453 | 9.1294 | Improved |
| 41 | -1.5509 | 8.5147 | Improved |
| 42 | -1.5348 | 8.1011 | Improved |
| 43 | -1.5226 | 7.2683 | Improved |
| 44 | -1.5304 | 6.3048 | Improved |
| 45 | -1.5215 | 5.4685 | Improved |
| 46 | -1.5153 | 5.0598 | Improved |
| 47 | -1.5190 | 4.2742 | Improved |
| 48 | -1.5274 | 3.8472 | Improved |
| 49 | -1.5252 | 3.5929 | Improved |
| 50 | -1.5197 | 2.7594 | Improved |
| 51 | -1.4815 | 2.4836 | Improved |
| 52 | -1.4952 | 2.1473 | Improved |
| 53 | -1.4848 | 2.3893 | No improvement 1/15 |
| 54 | -1.5004 | 2.5557 | No improvement 2/15 |
| 55 | -1.4924 | 2.3630 | No improvement 3/15 |
| 56 | -1.5031 | 2.5463 | No improvement 4/15 |
| 57 | -1.4700 | 2.0038 | Improved |
| 58 | -1.4867 | 2.0144 | No improvement 1/15 |
| 59 | -1.4835 | 2.3078 | No improvement 2/15 |
| 60 | -1.4726 | 2.1643 | No improvement 3/15 |
| 61 | -1.4633 | 1.5850 | Improved |
| 62 | -1.4578 | 1.6066 | No improvement 1/15 |
| 63 | -1.4418 | 1.4517 | Improved |
| 64 | -1.4327 | 1.8084 | No improvement 1/15 |
| 65 | -1.4612 | 2.0603 | No improvement 2/15 |
| 66 | -1.4501 | 1.6850 | No improvement 3/15 |
| 67 | -1.4291 | 1.9905 | No improvement 4/15 |
| 68 | -1.4500 | 1.9690 | No improvement 5/15 |
| 69 | -1.4018 | 1.5521 | No improvement 6/15 |
| 70 | -1.4374 | **1.2388** | **Improved / Best** |

Final GAN result:

```text
Best epoch: 70
Best generator loss: 1.2388
```

The full 70-epoch run therefore did not trigger early stopping because improvement continued sufficiently often.

---

# 32. V3 GAN Training Interpretation

### Symptom

Early training showed generator-loss increases and periods with no improvement.

Example:

```text
Epoch 1: 30.6199
Epoch 4: 32.8813
Epoch 11: 30.8113
```

### Diagnosis

Adversarial training is not expected to behave like ordinary supervised reconstruction training where the loss should monotonically decrease.

The WGAN-GP training log shows alternating periods of:

- no improvement;
- recovery;
- substantial generator-loss reduction.

### Fix / Training Mechanism

A patience-based early-stopping mechanism was implemented:

```text
patience = 15 epochs
minimum improvement = 0.001
```

The best generator/critic checkpoint was saved whenever the generator loss improved enough.

### Outcome

The model continued improving after several temporary plateaus and ultimately reached its best recorded generator loss at:

```text
Epoch 70
Generator loss = 1.2388
```

This demonstrates why terminating the run during the early plateau would have been premature.

---

# 33. V3 Stage 2 — Encoder Training

After WGAN-GP training:

```text
Generator → frozen
Critic → frozen
Encoder → trainable
```

The encoder learns:

```text
x → E(x) → z
```

and the generator reconstructs:

```text
x̂ = G(E(x))
```

The encoder objective combines:

```text
Lreconstruction
```

and:

```text
Lfeature
```

using:

```text
Lencoder = 0.9 Lreconstruction + 0.1 Lfeature
```

Both terms use L1 loss.

---

# 34. V3 Encoder Training — Complete Log

| Epoch | Average encoder loss |
|---:|---:|
| 1 | 0.15869 |
| 2 | 0.13258 |
| 3 | 0.12241 |
| 4 | 0.11795 |
| 5 | 0.10852 |
| 6 | 0.10348 |
| 7 | 0.10122 |
| 8 | 0.09816 |
| 9 | 0.09682 |
| 10 | 0.09476 |
| 11 | 0.09400 |
| 12 | 0.09256 |
| 13 | 0.09060 |
| 14 | 0.09081 |
| 15 | 0.08925 |
| 16 | 0.08888 |
| 17 | 0.08817 |
| 18 | 0.08765 |
| 19 | 0.08699 |
| 20 | 0.08668 |
| 21 | 0.08564 |
| 22 | 0.08566 |
| 23 | 0.08473 |
| 24 | 0.08455 |
| 25 | 0.08422 |
| 26 | 0.08371 |
| 27 | 0.08311 |
| 28 | 0.08308 |
| 29 | 0.08250 |
| 30 | 0.08183 |
| 31 | 0.08192 |
| 32 | 0.08146 |
| 33 | 0.08139 |
| 34 | 0.08082 |
| 35 | 0.08081 |
| 36 | 0.08000 |
| 37 | 0.08015 |
| 38 | 0.07985 |
| 39 | 0.07957 |
| 40 | 0.07938 |
| 41 | 0.07909 |
| 42 | 0.07869 |
| 43 | 0.07833 |
| 44 | 0.07832 |
| 45 | 0.07801 |
| 46 | 0.07787 |
| 47 | 0.07751 |
| 48 | 0.07727 |
| 49 | 0.07708 |
| 50 | **0.07685** |

The encoder loss reduced from:

```text
0.15869 → 0.07685
```

which is approximately a:

```text
51.6% reduction
```

across the 50-epoch run.

---

# 35. V3 Anomaly Score

The final f-AnoGAN score has two components.

## 35.1 Reconstruction error

For each image:

```text
Lrec(x) =
mean |x - G(E(x))|
```

This measures pixel-level disagreement.

## 35.2 Feature error

The critic's intermediate feature map is extracted for both:

```text
D(x)
```

and:

```text
D(G(E(x)))
```

The feature discrepancy is:

```text
Lfeat(x) =
mean |fD(x) - fD(G(E(x)))|
```

## 35.3 Final score

The submitted notebook uses:

```text
s(x) = 0.9 Lrec(x) + 0.1 Lfeat(x)
```

Therefore V3 considers two forms of disagreement:

```text
Pixel-level disagreement
        +
Critic feature-space disagreement
        ↓
Final anomaly score
```

This is the principal conceptual change from V1/V2.

---

# 36. V3 Normal-Only Threshold Calibration

Unlike V1 and V2, V3 explicitly creates a calibration subset from the normal training data.

Configuration:

```text
Calibration fraction = 15%
Threshold quantile = 90th percentile
Random seed = 42
```

The notebook reports:

```text
Calibration normal images = 3,000
```

and:

```text
threshold = 0.100311
```

The test labels are used only for final evaluation.

### Decision rule

```text
score > 0.100311
        ↓
    anomaly
```

This calibration procedure is methodologically different from V1/V2 and is an important part of the V3 experiment.

---

# 37. V3 Final Evaluation

The executed notebook reports:

| Metric | V3 f-AnoGAN |
|---|---:|
| Test normal | 2,100 |
| Test anomaly | 5,125 |
| Total test | 7,225 |
| AUROC | **0.9687** |
| AUPRC | **0.9863** |
| Accuracy | **0.9157** |
| Precision | **0.9498** |
| Recall | **0.9303** |
| F1 | **0.9400** |
| Threshold | **0.100311** |

Confusion matrix:

| Actual / Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | 1,848 | 252 |
| Anomaly | 357 | 4,768 |

This corresponds to approximately:

```text
Normal false-positive rate ≈ 12.0%
Anomaly detection rate ≈ 93.0%
```

---

# 38. V3 Example Scores

The executed notebook prints the lowest and highest five anomaly scores.

## Lowest five

| Score | Label | Example |
|---:|---:|---|
| 0.022192 | Normal | `ESP_015897_2655_RED-0052-fv.jpg` |
| 0.023367 | Normal | `ESP_015897_2655_RED-0052-r180.jpg` |
| 0.026315 | Normal | `ESP_015897_2655_RED-0052-fh.jpg` |
| 0.028825 | Normal | `PSP_003703_2035_RED-0075-r180.jpg` |
| 0.028984 | Normal | `ESP_015897_2655_RED-0035-r270.jpg` |

## Highest five

| Score | Label | Example |
|---:|---:|---|
| 0.431962 | Anomaly | `PSP_002776_2025_RED-0100-fh.jpg` |
| 0.383509 | Anomaly | `ESP_019263_2650_RED-0092.jpg` |
| 0.363343 | Anomaly | `ESP_019263_2650_RED-0092-r270.jpg` |
| 0.360204 | Anomaly | `ESP_036575_2075_RED-0142-fv.jpg` |
| 0.351711 | Anomaly | `ESP_019263_2650_RED-0089-r270.jpg` |

These examples demonstrate that the extremes of the learned score distribution align with the expected labels in the executed evaluation output.

---

# 39. V3 Explanation Mechanism

The competition asks not only for anomaly detection but also for an explanation/isolation mechanism.

The V3 notebook produces feature-discrepancy heatmaps by comparing:

```text
Original image
      ↓
Encoder
      ↓
Generator
      ↓
Reconstruction

and then:

Critic(original)
      vs
Critic(reconstruction)
```

The absolute feature difference is averaged across feature channels and resized to the image resolution.

The visualization contains:

1. Original image
2. Reconstruction
3. Feature-error heatmap
4. Feature-error overlay

---

# 40. V3 Heatmap Observation

The executed notebook generated eight feature-discrepancy visualizations.

All eight examples produced by that visualization cell were labelled:

```text
0 = Normal
```

Their **feature-only** scores were:

```text
0.2276
0.1169
0.1418
0.0908
0.0879
0.0788
0.1536
0.0809
```

### Critical interpretation

These values are **feature-only visualization scores**.

They are **not** the final combined anomaly scores:

```text
s(x) = 0.9 Lrec + 0.1 Lfeat
```

Therefore they must **not** be compared directly against:

```text
τ = 0.100311
```

This distinction is retained explicitly to avoid mixing two different quantities.

---

# 41. V2 Additional Qualitative/Category Analysis

The V2 notebook contains an additional screening experiment applied to anomaly images flagged above its threshold.

The measured count was:

```text
2,084 anomaly images flagged
out of
5,125 anomaly images
```

The notebook then computed two image-level features:

1. largest high-error connected component;
2. Laplacian variance.

It standardized these features and ran:

```text
KMeans(n_clusters=2)
```

The cluster centers were:

```text
Cluster 0:
blob size ≈ 640.22
laplacian variance ≈ 0.00851

Cluster 1:
blob size ≈ 1231.30
laplacian variance ≈ 0.00476
```

The larger-blob cluster was assigned to:

```text
Corrupted Crop
```

and the other cluster to:

```text
Sensor Artifact
```

The resulting **heuristic screening counts** were:

```text
Sensor Artifact: 885
Corrupted Crop: 1199
```

### Important qualification

These are **not ground-truth category labels**.

The notebook explicitly describes the procedure as a screening method based on line-dropout / edge-void characteristics.

It also states that this method:

```text
cannot detect Non-Martian Content
```

because non-Martian content requires a content-level distinction rather than simply identifying voids, seams, or line-dropout structures.

Therefore the changelog records these numbers as:

> **heuristic predicted categories among flagged V2 anomalies**

and not as ground-truth anomaly-category counts.

---

# 42. V2 → V3 Final Symptom → Diagnosis → Fix

| Stage | Symptom | Diagnosis | Fix |
|---|---|---|---|
| V2 | CVAE AUROC = 0.8599, below V1 CAE AUROC = 0.9163 | Probabilistic latent regularization improved the generative objective but did not improve anomaly ranking on this dataset | Change the anomaly representation rather than only adjusting the VAE |
| V3 | Reconstruction alone may miss structural/semantic discrepancies | A discriminator trained to distinguish generated and real normal imagery provides a learned feature representation | Use discriminator feature discrepancy in addition to reconstruction error |
| V3 | GAN training has temporary plateaus | Adversarial objectives need not decrease monotonically | WGAN-GP + 15-epoch patience + best-checkpoint saving |
| V3 | Direct latent inversion would otherwise require optimization per image | Per-image latent optimization is expensive | Train a dedicated encoder for feed-forward inversion |
| V3 | Need an unsupervised operating threshold | Threshold should be selected without using anomaly labels | Use 3,000 normal training calibration images and the 90th percentile |

---

# 43. Why f-AnoGAN Was the Final Direction

The V3 change was not simply “make the network bigger.”

The progression introduced increasingly different representations:

### V1 — Reconstruction

```text
Normal image
     ↓
CAE
     ↓
Reconstruction error
```

### V2 — Probabilistic reconstruction

```text
Normal image
     ↓
CVAE
     ↓
Probabilistic latent representation
     ↓
Reconstruction error
```

### V3 — Adversarial manifold + feature discrepancy

```text
Normal image
     ↓
WGAN-GP learns normal manifold
     ↓
Dedicated encoder
     ↓
Reconstruction
     +
Discriminator feature discrepancy
     ↓
Final anomaly score
```

This gives f-AnoGAN two complementary anomaly signals:

```text
1. Pixel disagreement
2. Feature-space disagreement
```

---

# 44. Final Empirical Model Ranking

The measured AUROC ranking is:

```text
1. V3 f-AnoGAN   0.9687
2. V1 CAE        0.9163
3. V2 CVAE       0.8599
```

Relative to V1:

```text
V3 improvement in AUROC
= 0.9687 - 0.9163
= 0.0524
```

Relative to V2:

```text
V3 improvement in AUROC
= 0.9687 - 0.8599
= 0.1088
```

The final V3 operating point also reports:

```text
Precision = 0.9498
Recall    = 0.9303
F1        = 0.9400
Accuracy  = 0.9157
AUPRC     = 0.9863
```

---

# 45. Important Comparison Caveat

The AUROC ranking is empirical, but this is **not a perfectly controlled architecture-only ablation**.

The versions differ in:

### Input resolution

```text
V1 = 224 × 224 RGB
V2 = 128 × 128 grayscale
V3 = 64 × 64 RGB
```

### Score definition

```text
V1 = local reconstruction error
V2 = local reconstruction error
V3 = 0.9 reconstruction + 0.1 discriminator feature error
```

### Threshold calibration

```text
V1 = 95th percentile of normal TEST scores
V2 = 95th percentile of normal TEST scores
V3 = 90th percentile of a normal TRAINING calibration split
```

Therefore:

> The final result demonstrates that the submitted V3 system achieved the highest measured anomaly-ranking performance, but the experiment should not be interpreted as a perfectly controlled comparison in which architecture is the only changing variable.

This is an important reproducibility point.

---

# 46. Complete Development Timeline

## Phase 1 — Baseline

```text
Problem
  ↓
Need an unsupervised anomaly detector
  ↓
Build CAE
  ↓
Train on normal images
  ↓
Use reconstruction error
  ↓
AUROC = 0.9163
```

### Lesson

A simple reconstruction model is already a strong baseline.

---

## Phase 2 — Probabilistic latent representation

```text
CAE
  ↓
Question:
Can a structured probabilistic latent space improve separation?
  ↓
CVAE
  ↓
30 epochs
  ↓
Reconstruction + KL
  ↓
AUROC = 0.8599
```

### Lesson

The additional probabilistic regularization did not improve the anomaly-ranking result.

---

## Phase 3 — Adversarial normal manifold

```text
CVAE
  ↓
Question:
Can a learned adversarial manifold and feature-space discrepancy
provide a stronger anomaly signal?
  ↓
WGAN-GP
  ↓
Dedicated encoder
  ↓
Reconstruction + feature error
  ↓
AUROC = 0.9687
AUPRC = 0.9863
```

### Lesson

The f-AnoGAN formulation produced the strongest measured result in the project.

---

# 47. Version-by-Version Symptom → Diagnosis → Fix Summary

## V1

### Symptom

No baseline existed for determining whether normal-only reconstruction could separate abnormal images.

### Diagnosis

A deterministic CAE can learn the dominant visual structure of normal imagery and produce an image-level reconstruction discrepancy.

### Fix

Build and train a CAE using only normal training images.

### Outcome

```text
AUROC = 0.9163
```

The baseline was strong enough to justify further representation experiments.

---

## V1 → V2

### Symptom

The CAE's latent space had no explicit probabilistic regularization.

### Diagnosis

A deterministic latent code does not explicitly constrain the encoded normal distribution.

### Fix

Introduce:

- μ
- log variance
- reparameterization
- KL divergence

through a CVAE.

### Outcome

```text
AUROC = 0.8599
```

The experiment did not improve empirical anomaly ranking.

---

## V2 → V3

### Symptom

The CVAE underperformed the simpler CAE despite substantial reductions in reconstruction loss.

### Diagnosis

The evidence indicates that improving the reconstruction/VAE objective was not sufficient to improve anomaly discrimination.

A broader anomaly signal was therefore needed.

### Fix

Introduce:

- WGAN-GP
- learned adversarial normal manifold
- dedicated encoder
- discriminator feature discrepancy
- combined reconstruction + feature score

### Outcome

```text
AUROC = 0.9687
AUPRC = 0.9863
F1 = 0.9400
```

---

# 48. Engineering / Training Fixes Within V3

The final version also contains several explicit engineering decisions.

## Fix 1 — WGAN-GP instead of ordinary GAN loss

### Problem

Ordinary GAN training can be difficult to stabilize.

### Fix

Use Wasserstein GAN with gradient penalty.

```text
Gradient penalty weight = 10
```

This provides a more controlled critic objective.

---

## Fix 2 — Multiple critic updates

### Problem

The critic needs sufficient optimization relative to the generator.

### Fix

Use:

```text
5 critic updates
per generator update
```

---

## Fix 3 — Best-checkpoint tracking

### Problem

GAN loss can fluctuate and later epochs are not automatically guaranteed to be the best model.

### Fix

Track the average generator loss and save the best checkpoint whenever:

```text
current_loss < best_loss - 0.001
```

---

## Fix 4 — Patience-based stopping

### Problem

Temporary plateaus should not immediately terminate adversarial training.

### Fix

Use:

```text
patience = 15
minimum improvement = 0.001
```

The run ultimately reached its best generator loss at epoch 70.

---

## Fix 5 — Dedicated encoder

### Problem

Directly optimizing a latent vector for every new image is computationally expensive.

### Fix

Train an encoder:

```text
image → latent vector
```

and use:

```text
image → encoder → generator
```

for fast inference.

---

## Fix 6 — Feature-aware anomaly score

### Problem

Pixel reconstruction alone can fail to represent higher-level structural discrepancies.

### Fix

Add discriminator feature discrepancy:

```text
score =
0.9 × reconstruction error
+
0.1 × feature error
```

---

# 49. Thresholding Evolution

Thresholding changed throughout the project.

| Version | Threshold source | Quantile | Threshold |
|---|---|---:|---:|
| V1 | Normal **test** scores | 95th percentile | 0.00823 |
| V2 | Normal **test** scores | 95th percentile | 0.04929 |
| V3 | Normal **training calibration split** | 90th percentile | 0.100311 |

### Interpretation

The V3 procedure is the cleanest of the three with respect to avoiding use of test-normal scores for threshold selection.

However, the threshold quantile itself changed from 95% to 90%, so thresholded metrics are not directly comparable as if the same operating point had been imposed on all three models.

For this reason, AUROC remains the primary model-comparison metric.

---

# 50. What the Final Results Demonstrate

The project establishes three useful experimental conclusions.

## Conclusion 1

A simple CAE is already capable of meaningful anomaly ranking:

```text
AUROC = 0.9163
```

## Conclusion 2

Adding a probabilistic latent bottleneck did not automatically improve anomaly detection:

```text
CVAE AUROC = 0.8599
```

which is below:

```text
CAE AUROC = 0.9163
```

## Conclusion 3

The f-AnoGAN system provided the strongest measured result:

```text
AUROC = 0.9687
AUPRC = 0.9863
```

with:

```text
Precision = 0.9498
Recall    = 0.9303
F1        = 0.9400
Accuracy  = 0.9157
```

---

# 51. Reproducibility Notes

## V1

```text
Input: 224 × 224 RGB
Batch: 64
Loss: MSE
Optimizer: Adam
LR: 1e-3
Weight decay: 1e-5
Epochs: 10
Score: local patch reconstruction maximum
Threshold: 95th percentile normal TEST scores
```

## V2

```text
Input: 128 × 128 grayscale
Batch: 128
Latent dimension: 64
Base channels: 32
Loss: MSE + 0.5 × KL
Optimizer: Adam
LR: 1e-3
Epochs: 30
Score: local patch reconstruction maximum
Threshold: 95th percentile normal TEST scores
```

## V3

```text
Input: 64 × 64 RGB
Batch: 32
Latent dimension: 128
Base channels: 64

WGAN-GP:
    Epochs: up to 70
    LR: 2e-4
    Adam betas: (0.0, 0.9)
    Critic steps: 5
    GP weight: 10
    Patience: 15
    Min improvement: 0.001

Encoder:
    Epochs: 50
    LR: 2e-4
    Adam betas: (0.5, 0.999)
    L1 reconstruction
    L1 feature loss
    Weight: 0.9 / 0.1

Calibration:
    Normal training split: 15%
    Calibration images: 3,000
    Quantile: 90%
    Seed: 42
    Threshold: 0.100311
```

---

# 52. Known Limitations and Open Improvements

The following are retained as future work rather than being represented as completed experiments.

## 52.1 Higher input resolution

V3 currently uses:

```text
64 × 64
```

Higher resolutions such as:

```text
128 × 128
256 × 256
```

could preserve finer Martian surface details.

---

## 52.2 Better threshold evaluation

A future controlled experiment should:

- create a dedicated normal calibration split;
- keep it separate from test normals;
- evaluate several quantiles;
- report precision/recall/F1 at each operating point.

---

## 52.3 Category-wise evaluation

The V2 clustering procedure provides heuristic categories for:

- Sensor Artifact
- Corrupted Crop

but not ground-truth category labels.

A future experiment should use verified category labels to calculate:

```text
per-category precision
per-category recall
per-category F1
```

---

## 52.4 Non-Martian content

The current V2 screening explicitly cannot identify Non-Martian Content.

A dedicated evaluation set would be needed for a reliable measurement of this category.

---

## 52.5 Combined anomaly scores

Future work can investigate:

```text
reconstruction score
+
feature score
+
latent-space score
```

and systematically optimize their weights.

---

## 52.6 Computational benchmarking

Future experiments should record:

- inference time per image;
- GPU memory;
- model parameter count;
- training time;
- throughput.

This would allow the performance gain of f-AnoGAN to be evaluated alongside computational cost.

---

# 53. Final Changelog Summary

```text
┌─────────────────────────────────────────────────────────────┐
│ V1 — CAE                                                    │
│                                                             │
│ Symptom: Need a strong unsupervised reconstruction baseline │
│ Diagnosis: Normal-only reconstruction can provide anomaly  │
│            separation                                       │
│ Fix: Train deterministic CAE on normal imagery              │
│ Result: AUROC = 0.9163                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ V2 — CVAE                                                   │
│                                                             │
│ Symptom: Deterministic latent space is unregularized        │
│ Diagnosis: Test probabilistic latent structure               │
│ Fix: Add μ, logvar, sampling and KL regularization           │
│ Result: AUROC = 0.8599                                      │
│ Lesson: Better VAE objective did not improve anomaly        │
│         ranking                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ V3 — f-AnoGAN                                               │
│                                                             │
│ Symptom: Reconstruction-based CVAE underperformed CAE        │
│ Diagnosis: Need a richer anomaly representation              │
│ Fix: WGAN-GP + encoder + feature discrepancy                 │
│ Result: AUROC = 0.9687, AUPRC = 0.9863                      │
└─────────────────────────────────────────────────────────────┘
```

## Final empirical ranking

```text
f-AnoGAN  ███████████████████  0.9687
CAE       █████████████████    0.9163
CVAE      ███████████████      0.8599
```

**Final selected model: V3 f-AnoGAN**

The selection is based on the strongest measured AUROC and the strongest reported thresholded metrics in the executed experiments, while acknowledging that the three versions use different resolutions, preprocessing, scoring definitions, and threshold-calibration procedures.

---

# 54. Source-of-Truth Rule for Future Edits

When modifying this changelog:

1. **Do not invent experimental values.**
2. Use the submitted notebooks as the primary source for code, hyperparameters, and recorded outputs.
3. Use the final report for the consolidated interpretation.
4. Clearly distinguish:
   - measured result,
   - implementation detail,
   - interpretation,
   - hypothesis,
   - future work.
5. Do not call heuristic category assignments ground truth.
6. Do not compare feature-only heatmap scores directly with the final combined f-AnoGAN threshold.
7. Do not describe V1/V2 test-derived thresholds as training-only calibration.
8. Do not claim that CVAE failure proves a specific pathology unless the notebook directly demonstrates it.
9. Preserve the exact **Symptom → Diagnosis → Fix** progression required by the competition.


---

**End of Changelog**
