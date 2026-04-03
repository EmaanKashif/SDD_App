import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🔬",
    layout="centered"
)

# ── constants ─────────────────────────────────────────────────
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

FULL_NAMES = {
    'nv':    'Melanocytic Nevus (Mole)',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma'
}

RISK_LEVEL = {
    'nv':    'Low risk',
    'mel':   'High risk',
    'bkl':   'Low risk',
    'bcc':   'High risk',
    'akiec': 'Moderate risk',
    'vasc':  'Low risk',
    'df':    'Low risk'
}

# ── load model (cached — only loads once) ─────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 7)
    )
    model.load_state_dict(
        torch.load('skin_model.pth', map_location=torch.device('cpu'))
    )
    model.eval()
    return model

# ── image transform ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── grad-cam ──────────────────────────────────────────────────
def generate_gradcam(model, input_tensor, pred_idx, original_img):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    img_array = np.array(original_img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    return visualization

# ── page ui ───────────────────────────────────────────────────
st.title("🔬 Skin Disease Detector")
st.markdown(
    "Upload a dermoscopy image to get an AI-powered prediction "
    "with Grad-CAM explainability."
)
st.markdown("---")

st.warning(
    "⚠️ For educational purposes only. "
    "Not a substitute for professional medical advice."
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    st.markdown("### Uploaded image")
    st.image(image, width=300)

    with st.spinner("Analysing..."):
        model = load_model()
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

    pred_class = CLASS_NAMES[pred_idx]

    # ── results ───────────────────────────────────────────
    st.markdown("### Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Diagnosis", pred_class.upper())
    col2.metric("Confidence", f"{confidence*100:.1f}%")
    col3.metric("Risk", RISK_LEVEL[pred_class])

    st.info(f"**Full name:** {FULL_NAMES[pred_class]}")

    # ── confidence chart ──────────────────────────────────
    st.markdown("### Confidence per class")
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [
        '#1D9E75' if i == pred_idx else '#B4B2A9'
        for i in range(len(CLASS_NAMES))
    ]
    bars = ax.barh(CLASS_NAMES, probs.numpy(), color=colors)
    ax.set_xlabel('Confidence')
    ax.set_xlim(0, 1)
    for bar, prob in zip(bars, probs.numpy()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{prob*100:.1f}%',
            va='center', fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── grad-cam ──────────────────────────────────────────
    st.markdown("### Grad-CAM — where the model looked")
    st.caption("Red/yellow = regions the model focused on most.")

    with st.spinner("Generating heatmap..."):
        visualization = generate_gradcam(
            model, input_tensor, pred_idx, image
        )

    col1, col2 = st.columns(2)
    col1.image(image.resize((224, 224)), caption="Original")
    col2.image(visualization, caption="Grad-CAM overlay")


from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="https://huggingface.co/EmaanK/SKD",  # 🔴 change this
        filename="skin_model.pth"  # ✅ your actual file name
    )
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

model = load_model()

 # ── model info ────────────────────────────────────────
st.markdown("---")
with st.expander("About this model"):
        st.markdown("""
**Architecture:** EfficientNet-B3 (pretrained on ImageNet, fine-tuned on HAM10000)

**Dataset:** HAM10000 — 10,015 dermoscopy images across 7 skin disease classes

**Performance:**
- Test accuracy: 77.7%
- Weighted F1 score: 0.792
- Macro F1 score: 0.655

**Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)

**Developed by:** DS undergraduate — University of Faisalabad, Pakistan
        """) 
