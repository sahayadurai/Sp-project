# VQA System

A reproducible visual question answering system trained from the `nlphuji/flickr30k` Hugging Face dataset. It does not use an all-in-one multimodal LLM. It combines frozen CLIP and DistilBERT encoders with learned cross-attention, a CLIP binary verifier, an object-color specialist, and a BLIP image-captioning component for open-ended one-line answers.

## Why caption-derived QA?

Flickr30k provides images and natural-language captions, not human-written VQA pairs. `src/qa_generation.py` turns captions into transparent lexical, color, count, and presence questions. The README and dashboard expose this limitation so evaluation results are interpretable.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --max-images 10000 --epochs 5 --batch-size 64 --examples-per-image 8 --device mps --output artifacts/flickr_vqa_patch_attention_10000.pt
streamlit run app.py
```

The first run downloads the Flickr30k dataset and the CLIP/DistilBERT checkpoints from Hugging Face. The submitted checkpoint was trained on 10,000 images with Apple MPS. Questions containing `color` or `colour` use an object-specific visual specialist with foreground cropping.

The dashboard routes questions by answer shape: `What's in this picture?` uses BLIP to produce a one-line description; `Is it a man?` uses CLIP pairwise verification and returns a sentence; color questions use the object-color specialist; other short-label questions use the trained cross-attention classifier. BLIP is used as an image-captioning component, not an all-in-one conversational multimodal LLM.

## Run with Docker

The included image packages the Streamlit app and the trained 10,000-image checkpoint. Build and launch it from the project root:

```bash
docker compose up --build
```

Open <http://localhost:8503>. Stop it with `Ctrl+C`; run `docker compose down` to remove the container. The named `huggingface-cache` volume preserves downloaded CLIP, DistilBERT, and BLIP model files between container runs. The first request requiring BLIP downloads its model weights, so the container needs internet access initially.

The Docker image installs CPU-only PyTorch. Docker Desktop on macOS does not expose Apple MPS/Metal to Linux containers, so native execution with `.venv` is faster on this Mac. On a Linux host with NVIDIA Container Toolkit, adapt the image to a CUDA-enabled PyTorch wheel and configure GPU access; the current Docker setup prioritizes a portable CPU demo.

The main fusion model uses all 50 CLIP vision tokens (the CLS token plus 49 image patches) as queries and all 48 DistilBERT question tokens as keys and values. This is genuine multi-token cross-attention, unlike pooled-vector attention. The 10,000-image training command writes `artifacts/flickr_vqa_patch_attention_10000.pt`.

## Project layout

- `train.py`: loads Flickr30k, derives QA examples, caches multimodal embeddings, and trains the fusion/head layers.
- `src/model.py`: CLIP + DistilBERT encoders, cross-attention fusion, and checkpoint inference.
- `src/model.py`: includes the object-specific color-region specialist.
- `src/model.py`: routes open-ended, binary, color, and short-label questions to appropriate specialists.
- `src/qa_generation.py`: deterministic caption-to-QA supervision.
- `src/data.py`: dataset schema normalization.
- `app.py`: Streamlit upload/question/ranked-answer dashboard.

## Evaluation caveat

The trained classifier can only answer labels observed during caption-derived training. For a stronger exam system, extend the QA generator with Flickr30k Entities object phrases and relation templates, add a held-out image split, and report exact match plus top-3 accuracy.
