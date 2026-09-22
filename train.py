"""Train the answer head: python train.py --max-images 500"""

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
    parser.add_argument("--output", default="artifacts/flickr_vqa.pt")
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
    embeddings, targets = [], []
    record_images = {record["image_id"]: image_from_record(record) for record in records}
    for start in tqdm(range(0, len(examples), args.batch_size), desc="Encoding Flickr30k"):
        batch = examples[start : start + args.batch_size]
        with torch.inference_mode():
            embeddings.append(model.encode([record_images[item.image_id] for item in batch], [item.question for item in batch]).cpu())
        targets.extend(label_to_id[item.answer] for item in batch)
    x = torch.cat(embeddings)
    y = torch.tensor(targets)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.head.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model.head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(y):.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"answer_labels": labels, "head": model.head.state_dict(), "clip_name": "openai/clip-vit-base-patch32", "text_name": "distilbert-base-uncased"}, output)
    print(f"saved {output} with {len(labels)} answers and {len(examples)} QA examples")


if __name__ == "__main__":
    main()
