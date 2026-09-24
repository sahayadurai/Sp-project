"""Streamlit dashboard for Flickr30k VQA."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.model import FlickrVQA

st.set_page_config(page_title="VQA System", page_icon="◈", layout="wide")
st.markdown("""
<style>
:root { --ink:#182326; --mint:#d8f3dc; --coral:#f4845f; --paper:#fffaf2; }
.stApp { background: radial-gradient(circle at 90% 0%, #f9d5c1 0, transparent 35%), linear-gradient(135deg, #fffaf2 0%, #e9f5db 100%); color:var(--ink); }
.block-container { max-width: 1180px; padding-top: 2rem; }
.hero { padding: 1.8rem 2rem; border: 1px solid #18232622; border-radius: 18px; background: #fffaf2cc; box-shadow: 8px 8px 0 #18232618; }
.hero h1 { font-family: Georgia, serif; font-size: 3rem; margin: 0; letter-spacing: 0; }
.hero p { font-size: 1.05rem; max-width: 760px; }
.answer { padding: 1rem 1.2rem; border-left: 6px solid var(--coral); background: #ffffffbb; border-radius: 8px; margin: .5rem 0; }
.answer strong { font-size: 1.3rem; }
.stFileUploader button { background: #fffaf2 !important; color: #182326 !important; border: 1px solid #182326 !important; font-weight: 700 !important; }
.stFileUploader button:hover { background: #f9d5c1 !important; color: #182326 !important; border-color: #182326 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>VQA System</h1><p>A compact, inspectable vision question answering system. Frozen CLIP and DistilBERT encoders meet in a learned cross-attention fusion block, without an all-in-one multimodal LLM.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Model status")
    checkpoint = st.text_input("Checkpoint", "artifacts/flickr_vqa_1000.pt")
    device = st.selectbox("Runtime", ["cpu", "cuda"], index=0)
    st.caption("Train with: python train.py --max-images 1000")

@st.cache_resource(show_spinner="Loading vision and language encoders...")
def load_model(path: str, runtime: str):
    return FlickrVQA.load_checkpoint(path, runtime if runtime == "cuda" else "cpu")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1 · Add an image")
    uploaded = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
    image = Image.open(uploaded).convert("RGB") if uploaded else None
    if image:
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to begin.")
with right:
    st.subheader("2 · Ask a question")
    question = st.text_input("Question", placeholder="What color is the dog?")
    ask = st.button("Get answer", type="primary", use_container_width=True)
    if ask:
        if image is None or not question.strip():
            st.warning("Provide both an image and a question.")
        elif not Path(checkpoint).exists():
            st.error("No checkpoint found. Run the training command in the sidebar first.")
        else:
            try:
                model = load_model(checkpoint, device)
                predictions = model.predict(image, question.strip())
                st.markdown("#### Ranked answers")
                for rank, (answer, confidence) in enumerate(predictions, 1):
                    st.markdown(f'<div class="answer"><strong>{rank}. {answer}</strong><br><small>model confidence {confidence:.1%}</small></div>', unsafe_allow_html=True)
            except Exception as error:
                st.exception(error)

with st.expander("System details and fusion method"):
    details = load_model(checkpoint, device).fusion_summary() if Path(checkpoint).exists() else {
        "image_encoder": "Frozen CLIP ViT-B/32",
        "question_encoder": "Frozen DistilBERT",
        "fusion": "Image query attends to question key/value",
        "trainable": "Projection layers, cross-attention, and answer head",
    }
    detail_columns = st.columns(len(details))
    for column, (label, value) in zip(detail_columns, details.items()):
        column.metric(label.replace("_", " ").title(), value)

with st.expander("Suggested evaluation questions"):
    st.write("Try object, color, presence, and count questions on images outside Flickr30k.")
    st.code("What color is the dog?\nWhat color is the ball?\nIs there a person?\nWhat is the main object?")

st.divider()
st.caption("Research note · color questions use an object-specific visual specialist; other questions use the cross-attention classifier. Flickr30k supplies images and captions rather than native VQA pairs.")
