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

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── fanogan_model.pth.zip
│
├── v1-auto-encoder.ipynb
├── v2-variational-autencoderp.ipynb
├── v3-f-AnoGAN.ipynb
├── test+train/
│   ├── train/
│   │   └── normal/
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

# Streamlit Dashboard

The final f-AnoGAN model is deployed through an interactive **Streamlit dashboard**.

The dashboard allows a user to upload a Mars surface image and run the trained V3 model for anomaly detection.

## Streamlit Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application containing the f-AnoGAN inference pipeline and dashboard interface |
| `requirements.txt` | Python dependencies required to run the dashboard |
| `fanogan_model.pth.zip` | Trained f-AnoGAN checkpoint used by the application |

## Dashboard Workflow

```text
Upload Image
     │
     ▼
Convert to RGB
     │
     ▼
Resize to 64×64
     │
     ▼
Normalize to [-1, 1]
     │
     ▼
Dedicated Encoder
     │
     ▼
Latent Representation
     │
     ▼
Generator
     │
     ▼
Reconstruction
     │
     ├──────────────────────┐
     ▼                      ▼
Reconstruction Error   Critic Feature Error
     │                      │
     └──────────┬───────────┘
                ▼
       0.9 × Rec + 0.1 × Feature
                │
                ▼
          Anomaly Score
                │
                ▼
       Threshold = 0.100311
                │
          ┌─────┴─────┐
          ▼           ▼
       NORMAL      ANOMALY
```

## Running the Dashboard Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Make sure the model checkpoint is present

The following files should be in the project root:

```text
app.py
requirements.txt
fanogan_model.pth.zip
```

### 5. Start Streamlit

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

Open the address shown by Streamlit in your browser.

## Using the Dashboard

1. Start the dashboard with `streamlit run app.py`.
2. Upload a JPG, JPEG, or PNG image.
3. The application converts the image to RGB.
4. The image is resized to `64×64`.
5. The image is normalized to `[-1, 1]`.
6. The encoder generates the latent representation.
7. The generator creates the reconstruction.
8. Reconstruction error and critic feature error are calculated.
9. The final anomaly score is calculated using:

```text
Score = 0.9 × Reconstruction Error
      + 0.1 × Critic Feature Error
```

10. The score is compared with the calibrated threshold:

```text
0.100311
```

The dashboard then presents the anomaly assessment and inference information.

> **Confidence note:** A confidence or decision-margin indicator should be interpreted as distance from the decision threshold, not as a calibrated probability.

## Testing the Dashboard

The dashboard can be tested using the existing test dataset:

```text
test+train/
└── test/
    ├── normal/
    └── anomaly_real/
```

### Normal image

Select an image from:

```text
test+train/test/normal/
```

and run it through the dashboard.

### Anomaly image

Select an image from:

```text
test+train/test/anomaly_real/
```

and run it through the dashboard.

The classification rule is:

```text
Score < 0.100311  → Normal
Score ≥ 0.100311  → Anomaly
```

## Troubleshooting

### Model checkpoint not found

Make sure `fanogan_model.pth.zip` is in the same project directory as `app.py`, and run Streamlit from the repository root.

### Missing Python packages

Activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

### Streamlit does not start

Try:

```bash
python -m streamlit run app.py
```

and make sure the virtual environment is activated.

### Checkpoint loading error

Make sure `fanogan_model.pth.zip` is the checkpoint corresponding to the final V3 f-AnoGAN implementation in `v3-f-AnoGAN.ipynb`.

---

## Documentation

- 📘 [`changelog.md`](changelog.md) — complete experiment evolution
- 📄 [`report.pdf`](report.pdf) — full technical report
