# HiRISE Anomaly Detection — ASTRA

**NSSC 2026 · Data Analytics Case Competition · Astronomy Club, IIT (BHU) Varanasi**

Unsupervised deep-generative anomaly detection for **HiRISE Mars imagery**, trained on normal imagery and evaluated for structural anomalies.

> **Final model: f-AnoGAN — AUROC 0.9687**

---

## Model Evolution

```text
V1 — CAE
   ↓
V2 — CVAE
   ↓
V3 — f-AnoGAN 🏆
```

| Version | Model | Input | AUROC | Core idea |
|---|---|---|---:|---|
| **V1** | CAE | 224×224 RGB | **0.9163** | Reconstruction error |
| **V2** | CVAE | 128×128 Grayscale | **0.8599** | Probabilistic latent space + KL |
| **V3** | f-AnoGAN | 64×64 RGB | **0.9687** | Reconstruction + critic feature discrepancy |

The development of every version follows **Symptom → Diagnosis → Fix** and is documented in [`changelog.md`](changelog.md).

---

## Final f-AnoGAN Performance

**20,000 normal training images · 2,100 normal test images · 5,125 anomaly test images**

| Metric | Score |
|---|---:|
| **AUROC** | **0.9687** |
| **AUPRC** | **0.9863** |
| Accuracy | 0.9157 |
| Precision | 0.9498 |
| Recall | 0.9303 |
| F1 | 0.9400 |
| Threshold | 0.100311 |

### Confusion Matrix

| Actual \ Predicted | Normal | Anomaly |
|---|---:|---:|
| **Normal** | 1848 | 252 |
| **Anomaly** | 357 | 4768 |

---

## f-AnoGAN Pipeline

```text
Normal images
     │
     ▼
   WGAN-GP
  ┌───────┐
  │ G + D │
  └───┬───┘
      │
      ▼
Dedicated Encoder
      │
      ▼
     z → Generator → Reconstruction
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Reconstruction Error    Feature Error
          └─────────┬─────────┘
                    ▼
        0.9 × Rec + 0.1 × Feature
                    │
                    ▼
             Anomaly Score
```

The final score is explicitly:

```text
Score = 0.9 × Reconstruction Error
      + 0.1 × Critic Feature Error
```

The V3 threshold is calibrated using **3,000 normal training images** at the **90th percentile**, giving `0.100311`.

---

## Training Snapshot

### WGAN-GP

- 70 epochs completed
- LR: `2e-4`
- 5 critic updates / generator update
- Gradient penalty: `10`
- Patience: `15`
- Best generator loss: **1.2388**

### Encoder

- 50 epochs
- LR: `2e-4`
- L1 reconstruction + feature loss
- Weighting: `0.9 / 0.1`
- Loss: **0.15869 → 0.07685** (~51.6% reduction)

The training configuration and encoder progression are recorded in the changelog.

---

## Explainability

f-AnoGAN compares critic features of the original image and its reconstruction to produce a **feature-discrepancy heatmap**.

Eight feature-error visualizations were generated. Selected examples are included in the report for qualitative analysis.

---

## Repository Layout

```text
.
├── v1-auto-encoder.ipynb
├── v2-variational-autencoderp.ipynb
├── v3-f-AnoGAN.ipynb
├── test+train/
│   ├── train/normal/
│   └── test/
│       ├── normal/
│       └── anomaly_real/
├── report.pdf
├── changelog.md
└── README.md
```

The dataset is organized as normal training data with normal/anomaly test data, while anomaly labels are not used for generative-model fitting.

---

## Key Result

```text
f-AnoGAN   0.9687  🏆
CAE        0.9163
CVAE       0.8599
```

f-AnoGAN improved the measured AUROC by **0.0524** over the CAE and **0.1088** over the CVAE.

> **Comparison note:** V1, V2 and V3 use different resolutions, preprocessing, score definitions and threshold-calibration procedures. The ranking is therefore an empirical result, not a perfectly controlled architecture-only ablation.

---

## Documentation

- 📘 [`changelog.md`](changelog.md) — complete experiment evolution
- 📄 [`report.pdf`](report.pdf) — full technical report
