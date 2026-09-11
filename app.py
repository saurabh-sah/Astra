import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# Configuration — matches the trained V3 f-AnoGAN setup
# ============================================================
MODEL_PATH = Path("fanogan_model.pth.zip")
IMAGE_SIZE = 64
LATENT_DIM = 128
THRESHOLD = 0.100311

REC_WEIGHT = 0.9
FEATURE_WEIGHT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model definitions — must match the training notebook
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()

        # The trained critic exposes this feature map for f-AnoGAN.
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.critic = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, x, return_features=False):
        features = self.features(x)
        score = self.critic(features).view(x.size(0), -1)

        if return_features:
            return score, features

        return score


class Encoder(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.latent_layer = nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.features(x)
        z = self.latent_layer(x)
        return z.view(x.size(0), -1)


# ============================================================
# Load trained model
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put the model file next to app.py."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    generator = Generator(
        latent_dim=checkpoint.get("latent_dim", LATENT_DIM),
        image_channels=checkpoint.get("image_channels", 3),
    ).to(DEVICE)

    discriminator = Discriminator(
        image_channels=checkpoint.get("image_channels", 3)
    ).to(DEVICE)

    encoder = Encoder(
        image_channels=checkpoint.get("image_channels", 3),
        latent_dim=checkpoint.get("latent_dim", LATENT_DIM),
    ).to(DEVICE)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    generator.eval()
    discriminator.eval()
    encoder.eval()

    return generator, discriminator, encoder


# ============================================================
# Image preprocessing
# Matches V3:
# Resize -> RGB -> ToTensor -> Normalize(mean=.5, std=.5)
# ============================================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Training normalization: approximately [-1, 1]
    tensor = (tensor - 0.5) / 0.5

    return tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# Inference
# ============================================================
@torch.inference_mode()
def predict(image):
    generator, discriminator, encoder = load_model()

    x = preprocess_image(image)

    # Image -> latent representation -> reconstruction
    z = encoder(x).view(-1, LATENT_DIM, 1, 1)
    reconstruction = generator(z)

    # Reconstruction error
    reconstruction_error = torch.mean(
        torch.abs(x - reconstruction),
        dim=(1, 2, 3)
    )

    # Critic feature discrepancy
    _, original_features = discriminator(x, return_features=True)
    _, reconstructed_features = discriminator(
        reconstruction, return_features=True
    )

    feature_error = torch.mean(
        torch.abs(original_features - reconstructed_features),
        dim=(1, 2, 3)
    )

    # Final f-AnoGAN score
    anomaly_score = (
        REC_WEIGHT * reconstruction_error
        + FEATURE_WEIGHT * feature_error
    )

    score = float(anomaly_score.item())
    rec_error = float(reconstruction_error.item())
    feat_error = float(feature_error.item())

    is_anomaly = score >= THRESHOLD

    # This is a decision-margin confidence, NOT a calibrated probability.
    # It tells the user how far the score is from the learned threshold.
    margin = abs(score - THRESHOLD) / max(THRESHOLD, 1e-8)
    decision_confidence = min(99.0, 50.0 + 50.0 * margin)

    # Visualization: convert reconstruction back from [-1, 1] to [0, 1].
    recon = reconstruction.squeeze(0).detach().cpu()
    recon = (recon * 0.5 + 0.5).clamp(0, 1)
    recon = recon.permute(1, 2, 0).numpy()
    recon = (recon * 255).astype(np.uint8)
    reconstruction_image = Image.fromarray(recon)

    return {
        "score": score,
        "reconstruction_error": rec_error,
        "feature_error": feat_error,
        "is_anomaly": is_anomaly,
        "confidence": decision_confidence,
        "reconstruction": reconstruction_image,
    }


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="ASTRA | HiRISE Anomaly Detection",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #0b1020;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #111936, #172554);
            border: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
        }

        .hero p {
            margin-top: 0.5rem;
            color: #cbd5e1;
            font-size: 1.05rem;
        }

        .result-card {
            padding: 1.4rem;
            border-radius: 16px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.12);
            margin: 0.5rem 0 1rem 0;
        }

        .normal {
            background: rgba(34,197,94,0.12);
        }

        .anomaly {
            background: rgba(239,68,68,0.12);
        }

        .result-title {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .result-subtitle {
            color: #cbd5e1;
            margin-top: 0.3rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🔭 ASTRA — HiRISE Anomaly Detection</h1>
        <p>
            f-AnoGAN based screening of Mars surface imagery for structural anomalies.
            Upload a HiRISE-style surface image and inspect its anomaly score,
            decision, and reconstruction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model")

    st.markdown(
        f"""
        **Model:** f-AnoGAN  
        **Input:** {IMAGE_SIZE}×{IMAGE_SIZE} RGB  
        **Latent dimension:** {LATENT_DIM}  
        **Score threshold:** `{THRESHOLD:.6f}`  
        **Device:** `{DEVICE}`
        """
    )

    st.divider()

    st.subheader("How the score works")

    st.latex(
        r"S = 0.9R + 0.1F"
    )

    st.caption(
        "R = reconstruction error, F = critic feature discrepancy."
    )

    st.info(
        "The displayed confidence is a decision-margin indicator. "
        "It is not a calibrated probability."
    )

    st.divider()

    st.caption(
        "ASTRA · Astronomy Club, IIT (BHU) Varanasi · NSSC 2026"
    )


# Upload
st.subheader("1. Upload a Mars surface image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a HiRISE/Mars surface image similar to the training data.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="small-note">
            Upload an image to run the trained f-AnoGAN model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

# Original preview
st.subheader("2. Input")

left, right = st.columns([1.3, 1])

with left:
    st.image(
        image,
        caption=f"Uploaded image · {image.width} × {image.height}px",
        use_container_width=True,
    )

with right:
    st.markdown(
        """
        **Preprocessing**

        - Converted to RGB
        - Resized to 64 × 64
        - Normalized to approximately [-1, 1]
        - Passed through the trained encoder
        """
    )

# Run inference
if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
    with st.spinner("Running f-AnoGAN inference..."):
        try:
            result = predict(image)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    st.subheader("3. Result")

    if result["is_anomaly"]:
        st.markdown(
            f"""
            <div class="result-card anomaly">
                <div class="result-title">⚠️ ANOMALOUS MARS IMAGE</div>
                <div class="result-subtitle">
                    The anomaly score is above the calibrated threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card normal">
                <div class="result-title">✅ NORMAL / GOOD MARS IMAGE</div>
                <div class="result-subtitle">
                    The anomaly score is below the calibrated threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Anomaly Score", f"{result['score']:.6f}")

    with c2:
        st.metric("Threshold", f"{THRESHOLD:.6f}")

    with c3:
        st.metric("Decision Confidence", f"{result['confidence']:.1f}%")

    with c4:
        label = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        st.metric("Classification", label)

    st.progress(
        min(result["score"] / max(THRESHOLD * 2, 1e-8), 1.0),
        text="Relative anomaly-score level",
    )

    # Error decomposition
    st.subheader("4. Score Breakdown")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Reconstruction Error</b><br>
                <span style="font-size:1.5rem">{result['reconstruction_error']:.6f}</span>
                <br><span class="small-note">90% of final score</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Critic Feature Error</b><br>
                <span style="font-size:1.5rem">{result['feature_error']:.6f}</span>
                <br><span class="small-note">10% of final score</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reconstruction
    st.subheader("5. Reconstruction")

    r1, r2 = st.columns(2)

    with r1:
        st.image(
            image,
            caption="Original",
            use_container_width=True,
        )

    with r2:
        st.image(
            result["reconstruction"],
            caption="f-AnoGAN reconstruction",
            use_container_width=True,
        )

    st.caption(
        "Large discrepancies between the input and its reconstruction can "
        "contribute to a higher anomaly score. The critic feature discrepancy "
        "adds a learned feature-level signal."
    )

    # Technical details
    with st.expander("🔎 Technical details"):
        st.write(
            {
                "Model": "f-AnoGAN",
                "Input size": f"{IMAGE_SIZE} × {IMAGE_SIZE}",
                "Latent dimension": LATENT_DIM,
                "Reconstruction weight": REC_WEIGHT,
                "Feature weight": FEATURE_WEIGHT,
                "Threshold": THRESHOLD,
                "Anomaly score": result["score"],
                "Reconstruction error": result["reconstruction_error"],
                "Feature error": result["feature_error"],
                "Decision": "Anomaly" if result["is_anomaly"] else "Normal",
                "Device": str(DEVICE),
            }
        )
