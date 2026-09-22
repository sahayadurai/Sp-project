# Flickr30k VQA Lab

A reproducible visual question answering baseline trained from the `nlphuji/flickr30k` Hugging Face dataset. It does not use an all-in-one multimodal LLM. The model combines a frozen OpenAI CLIP image encoder with a frozen DistilBERT question encoder and trains only a small answer-classification head.

## Why caption-derived QA?

Flickr30k provides images and natural-language captions, not human-written VQA pairs. `src/qa_generation.py` turns captions into transparent lexical, color, count, and presence questions. The README and dashboard expose this limitation so evaluation results are interpretable.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --max-images 1000 --epochs 5
streamlit run app.py
```

The first run downloads the Flickr30k dataset and the CLIP/DistilBERT checkpoints from Hugging Face. For a laptop demo, start with `--max-images 100` and increase it for better coverage. Use `--device cuda` only on a CUDA-enabled machine; macOS uses CPU or can be adapted to MPS in `src/model.py`.

## Project layout

- `train.py`: loads Flickr30k, derives QA examples, caches multimodal embeddings, and trains the answer head.
- `src/model.py`: non-generative CLIP + DistilBERT architecture and checkpoint inference.
- `src/qa_generation.py`: deterministic caption-to-QA supervision.
- `src/data.py`: dataset schema normalization.
- `app.py`: Streamlit upload/question/ranked-answer dashboard.

## Evaluation caveat

The trained classifier can only answer labels observed during caption-derived training. For a stronger exam system, extend the QA generator with Flickr30k Entities object phrases and relation templates, add a held-out image split, and report exact match plus top-3 accuracy.
