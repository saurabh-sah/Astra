# HiRISE Anomaly Detection — ASTRA

**NSSC 2026 · Data Analytics Case Competition · Astronomy Club, IIT (BHU) Varanasi**

Unsupervised deep-generative anomaly detection for **HiRISE Mars imagery**, trained on normal imagery and evaluated for structural anomalies.

> **Final model: f-AnoGAN — AUROC 0.9687**

---

## Project Overview

The project explores three progressively improved deep-learning approaches for detecting anomalies in HiRISE Mars surface imagery:

```text
V1 — CAE
   ↓
V2 — CVAE
   ↓
V3 — f-AnoGAN 🏆
   ↓
Streamlit Deployment
