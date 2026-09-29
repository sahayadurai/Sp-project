"""Train the cross-attention VQA model: python train.py --max-images 10000 --device mps"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data import captions, image_from_record, load_flickr30k
from src.model import FlickrVQA, resolve_device
from src.qa_generation import build_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples-per-image", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", default="artifacts/flickr_vqa_patch_attention_10000.pt")
    args = parser.parse_args()
    device = resolve_device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps"))
    print(f"training device: {device}")

    dataset = load_flickr30k(args.split, streaming=True)
    records = [
        {"image_id": str(index), "caption": captions(row), "image": row["image"]}
        for index, row in enumerate(islice(dataset, args.max_images))
    ]
    examples = build_examples(records)
    if args.examples_per_image > 0:
        examples_by_image = {}
        for example in examples:
            examples_by_image.setdefault(example.image_id, []).append(example)
        examples = [example for image_examples in examples_by_image.values() for example in image_examples[:args.examples_per_image]]
    counts = Counter(example.answer for example in examples)
    labels = sorted(answer for answer, count in counts.items() if count >= 2)
    examples = [example for example in examples if example.answer in labels]
    if not examples:
        raise RuntimeError("No synthetic QA examples were produced. Check the dataset schema.")
    label_to_id = {label: index for index, label in enumerate(labels)}

    model = FlickrVQA(labels).to(device)
    model.eval()
    image_embeddings, text_embeddings, text_masks, targets = [], [], [], []
    record_images = {record["image_id"]: image_from_record(record) for record in records}
    for start in tqdm(range(0, len(examples), args.batch_size), desc="Encoding Flickr30k"):
        batch = examples[start : start + args.batch_size]
        with torch.inference_mode():
            image_batch, text_batch, mask_batch = model.encode_modalities(
                [record_images[item.image_id] for item in batch],
                [item.question for item in batch],
            )
            image_embeddings.append(image_batch.cpu())
            text_embeddings.append(text_batch.cpu())
            text_masks.append(mask_batch.cpu())
        targets.extend(label_to_id[item.answer] for item in batch)
    image_x = torch.cat(image_embeddings)
    text_x = torch.cat(text_embeddings)
    mask_x = torch.cat(text_masks)
    y = torch.tensor(targets)
    loader = DataLoader(TensorDataset(image_x, text_x, mask_x, y), batch_size=args.batch_size, shuffle=True)
    trainable = list(model.image_projection.parameters()) + list(model.text_projection.parameters()) + list(model.cross_attention.parameters()) + list(model.fusion_norm.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_image, batch_text, batch_mask, batch_y in loader:
            batch_image = batch_image.to(device)
            batch_text = batch_text.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model.forward_features(batch_image, batch_text, batch_mask), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(y):.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "answer_labels": labels,
        "image_projection": model.image_projection.state_dict(),
        "text_projection": model.text_projection.state_dict(),
        "cross_attention": model.cross_attention.state_dict(),
        "fusion_norm": model.fusion_norm.state_dict(),
        "head": model.head.state_dict(),
        "clip_name": "openai/clip-vit-base-patch32",
        "text_name": "distilbert-base-uncased",
    }, output)
    print(f"saved {output} with {len(labels)} answers and {len(examples)} QA examples")


if __name__ == "__main__":
    main()
