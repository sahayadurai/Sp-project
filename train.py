"""Train the cross-attention VQA model: python train.py --max-images 1000"""

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
from src.model import FlickrVQA
from src.qa_generation import build_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="artifacts/flickr_vqa_1000.pt")
    args = parser.parse_args()

    dataset = load_flickr30k(args.split, streaming=True)
    records = [
        {"image_id": str(index), "caption": captions(row), "image": row["image"]}
        for index, row in enumerate(islice(dataset, args.max_images))
    ]
    examples = build_examples(records)
    counts = Counter(example.answer for example in examples)
    labels = sorted(answer for answer, count in counts.items() if count >= 2)
    examples = [example for example in examples if example.answer in labels]
    if not examples:
        raise RuntimeError("No synthetic QA examples were produced. Check the dataset schema.")
    label_to_id = {label: index for index, label in enumerate(labels)}

    model = FlickrVQA(labels)
    model.eval()
    image_embeddings, text_embeddings, targets = [], [], []
    record_images = {record["image_id"]: image_from_record(record) for record in records}
    for start in tqdm(range(0, len(examples), args.batch_size), desc="Encoding Flickr30k"):
        batch = examples[start : start + args.batch_size]
        with torch.inference_mode():
            image_batch, text_batch = model.encode_modalities(
                [record_images[item.image_id] for item in batch],
                [item.question for item in batch],
            )
            image_embeddings.append(image_batch.cpu())
            text_embeddings.append(text_batch.cpu())
        targets.extend(label_to_id[item.answer] for item in batch)
    image_x = torch.cat(image_embeddings)
    text_x = torch.cat(text_embeddings)
    y = torch.tensor(targets)
    loader = DataLoader(TensorDataset(image_x, text_x, y), batch_size=args.batch_size, shuffle=True)
    trainable = list(model.image_projection.parameters()) + list(model.text_projection.parameters()) + list(model.cross_attention.parameters()) + list(model.fusion_norm.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_image, batch_text, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model.forward_features(batch_image, batch_text), batch_y)
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
