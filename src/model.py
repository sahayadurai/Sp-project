"""Non-generative CLIP + text encoder VQA model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


class FlickrVQA(nn.Module):
    def __init__(self, answer_labels: list[str], clip_name: str = "openai/clip-vit-base-patch32", text_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.answer_labels = answer_labels
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_name)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        image_size = self.clip.config.projection_dim
        text_size = self.text_encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(image_size + text_size),
            nn.Linear(image_size + text_size, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, len(answer_labels)),
        )

    def encode(self, images: list[Image.Image], questions: list[str]) -> torch.Tensor:
        device = next(self.head.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_features = self.clip.get_image_features(**inputs)
        tokens = self.tokenizer(questions, padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
        text_features = self.text_encoder(**tokens).last_hidden_state[:, 0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return torch.cat([image_features, text_features], dim=-1)

    def forward(self, images: list[Image.Image], questions: list[str]) -> torch.Tensor:
        return self.head(self.encode(images, questions))

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> "FlickrVQA":
        bundle = torch.load(path, map_location=device)
        model = cls(bundle["answer_labels"], bundle.get("clip_name", "openai/clip-vit-base-patch32"), bundle.get("text_name", "distilbert-base-uncased"))
        model.head.load_state_dict(bundle["head"])
        model.to(device).eval()
        return model

    def predict(self, image: Image.Image, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        with torch.inference_mode():
            probabilities = self(image, [question]).softmax(-1)[0]
        values, indices = probabilities.topk(min(top_k, len(self.answer_labels)))
        return [(self.answer_labels[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())]
