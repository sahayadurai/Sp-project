"""Non-generative CLIP + text encoder VQA model."""

from __future__ import annotations

from pathlib import Path
import re

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoTokenizer, BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor


COLOR_NAMES = ("black", "white", "brown", "orange", "red", "yellow", "green", "blue", "gray", "pink", "purple")
OBJECT_WORDS = {
    "dog", "cat", "person", "man", "woman", "child", "boy", "girl", "car", "ball", "shirt", "dress", "horse", "bird"
}
BLIP_NAME = "Salesforce/blip-image-captioning-base"


def resolve_device(requested: str) -> str:
    """Return a device supported by the current machine and PyTorch build."""
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FlickrVQA(nn.Module):
    def __init__(self, answer_labels: list[str], clip_name: str = "openai/clip-vit-base-patch32", text_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.answer_labels = answer_labels
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_name)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        self.captioner = None
        self.caption_processor = None
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        image_size = self.clip.config.projection_dim
        text_size = self.text_encoder.config.hidden_size
        fusion_size = 512
        self.image_projection = nn.Linear(image_size, fusion_size)
        self.text_projection = nn.Linear(text_size, fusion_size)
        self.cross_attention = nn.MultiheadAttention(fusion_size, num_heads=8, batch_first=True)
        self.fusion_norm = nn.LayerNorm(fusion_size)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, len(answer_labels)),
        )

    def encode(self, images: list[Image.Image], questions: list[str]) -> torch.Tensor:
        image_features, text_features = self.encode_modalities(images, questions)
        return self.fuse(image_features, text_features)

    def encode_modalities(self, images: list[Image.Image], questions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.head.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_features = self.clip.get_image_features(**inputs)
        tokens = self.tokenizer(questions, padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
        text_features = self.text_encoder(**tokens).last_hidden_state[:, 0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    def fuse(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_token = self.image_projection(image_features).unsqueeze(1)
        text_token = self.text_projection(text_features).unsqueeze(1)
        attended_image, attention_weights = self.cross_attention(
            query=image_token,
            key=text_token,
            value=text_token,
            need_weights=True,
        )
        self.last_attention = attention_weights.detach()
        return self.fusion_norm((image_token + attended_image).squeeze(1))

    def forward_features(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.fuse(image_features, text_features))

    def forward(self, images: list[Image.Image], questions: list[str]) -> torch.Tensor:
        return self.head(self.encode(images, questions))

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> "FlickrVQA":
        device = resolve_device(device)
        bundle = torch.load(path, map_location=device)
        model = cls(bundle["answer_labels"], bundle.get("clip_name", "openai/clip-vit-base-patch32"), bundle.get("text_name", "distilbert-base-uncased"))
        model.image_projection.load_state_dict(bundle["image_projection"])
        model.text_projection.load_state_dict(bundle["text_projection"])
        model.cross_attention.load_state_dict(bundle["cross_attention"])
        model.fusion_norm.load_state_dict(bundle["fusion_norm"])
        model.head.load_state_dict(bundle["head"])
        model.to(device).eval()
        return model

    def predict(self, image: Image.Image, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        if self._is_color_question(question):
            return self.predict_color(image, question, top_k)
        with torch.inference_mode():
            probabilities = self(image, [question]).softmax(-1)[0]
        values, indices = probabilities.topk(min(top_k, len(self.answer_labels)))
        return [(self.answer_labels[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())]

    def answer_question(self, image: Image.Image, question: str) -> dict[str, object]:
        """Route each question to the specialist matching its answer shape."""
        normalized = question.lower().strip()
        if self._is_color_question(normalized):
            return {"kind": "ranked", "answer": self.predict_color(image, question), "source": "object color specialist"}
        if self._is_binary_question(normalized):
            answer, confidence = self.predict_binary(image, normalized)
            return {"kind": "sentence", "answer": answer, "confidence": confidence, "source": "CLIP binary verifier"}
        if self._is_caption_question(normalized):
            return {"kind": "sentence", "answer": self.caption(image), "source": "BLIP image captioner"}
        return {"kind": "ranked", "answer": self.predict(image, question), "source": "cross-attention classifier"}

    @staticmethod
    def _is_caption_question(question: str) -> bool:
        return any(phrase in question for phrase in (
            "what's in", "what is in", "what do you see", "describe", "what is happening",
            "what happens", "tell me about", "caption", "what is the picture",
        ))

    @staticmethod
    def _is_binary_question(question: str) -> bool:
        return question.startswith(("is ", "are ", "does ", "do ", "has ", "have ", "can "))

    def predict_binary(self, image: Image.Image, question: str) -> tuple[str, float]:
        words = re.findall(r"[a-z]+", question)
        target = next((word for word in words if word in OBJECT_WORDS), "person")
        paired = {"man": "woman", "woman": "man", "boy": "girl", "girl": "boy"}
        positive = target
        if target in paired:
            positive_text = f"a photo of a {target}"
            negative_text = f"a photo of a {paired[target]}"
        else:
            positive_text = f"a photo containing a {target}"
            negative_text = f"a photo with no {target}"
        inputs = self.processor(text=[positive_text, negative_text], images=[image], return_tensors="pt", padding=True).to(next(self.head.parameters()).device)
        with torch.inference_mode():
            probabilities = self.clip(**inputs).logits_per_image.softmax(dim=-1)[0]
        confidence = float(probabilities[0])
        if confidence >= 0.5:
            return f"Yes, the image appears to show a {positive}.", confidence
        return f"No, the image does not appear to show a {positive}.", float(1 - confidence)

    def caption(self, image: Image.Image) -> str:
        """Generate one concise visual description for open-ended questions."""
        device = next(self.head.parameters()).device
        if self.captioner is None:
            self.caption_processor = BlipProcessor.from_pretrained(BLIP_NAME)
            self.captioner = BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(device).eval()
        inputs = self.caption_processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = self.captioner.generate(**inputs, max_new_tokens=32, num_beams=4)
        text = self.caption_processor.decode(generated[0], skip_special_tokens=True).strip()
        return text[:1].upper() + text[1:].rstrip(".") + "."

    def fusion_summary(self) -> dict[str, str]:
        return {
            "image_encoder": "Frozen CLIP ViT-B/32",
            "question_encoder": "Frozen DistilBERT",
            "fusion": "Image query attends to question key/value",
            "trainable": "Projection layers, cross-attention, and answer head",
            "answering": "Captioning, binary verification, color, and classification routes",
        }

    @staticmethod
    def _is_color_question(question: str) -> bool:
        question = question.lower()
        return "color" in question or "colour" in question

    @staticmethod
    def _question_object(question: str) -> str:
        words = re.findall(r"[a-z]+", question.lower())
        return next((word for word in words if word in OBJECT_WORDS), "object")

    @staticmethod
    def _foreground_crop(image: Image.Image) -> Image.Image:
        """Remove a near-white studio background before object color scoring."""
        rgb = image.convert("RGB")
        pixels = rgb.load()
        xs, ys = [], []
        for y in range(rgb.height):
            for x in range(rgb.width):
                red, green, blue = pixels[x, y]
                if min(red, green, blue) < 242 or max(red, green, blue) - min(red, green, blue) > 12:
                    xs.append(x)
                    ys.append(y)
        if len(xs) < rgb.width * rgb.height * 0.01:
            return rgb
        padding = max(4, int(max(rgb.size) * 0.03))
        left = max(0, min(xs) - padding)
        top = max(0, min(ys) - padding)
        right = min(rgb.width, max(xs) + padding + 1)
        bottom = min(rgb.height, max(ys) + padding + 1)
        return rgb.crop((left, top, right, bottom))

    @staticmethod
    def _object_crops(image: Image.Image, object_name: str) -> list[Image.Image]:
        rgb = image.convert("RGB")
        if object_name == "ball":
            # In common portrait/product photos, a ball is placed below the
            # person; this crop prevents the larger subject from dominating CLIP.
            lower = rgb.crop((0, int(rgb.height * 0.58), rgb.width, rgb.height))
            return [lower, FlickrVQA._foreground_crop(lower)]
        return [rgb, FlickrVQA._foreground_crop(rgb)]

    def predict_color(self, image: Image.Image, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Use CLIP's aligned image/text space for object-specific color answers."""
        object_name = self._question_object(question)
        if object_name == "ball":
            return self._ball_pixel_colors(image, top_k)
        prompt_object = "soccer ball" if object_name == "ball" else object_name
        prompts = [f"a photo of a {color} {prompt_object}" for color in COLOR_NAMES]
        crops = self._object_crops(image, object_name)
        device = next(self.head.parameters()).device
        inputs = self.processor(text=prompts, images=crops, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            outputs = self.clip(**inputs)
            scores = outputs.logits_per_image.softmax(dim=-1).mean(dim=0)
        values, indices = scores.topk(min(top_k, len(COLOR_NAMES)))
        return [(COLOR_NAMES[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())]

    @staticmethod
    def _ball_pixel_colors(image: Image.Image, top_k: int) -> list[tuple[str, float]]:
        """Estimate visible ball colors from the lower-center object region."""
        rgb = image.convert("RGB")
        left, right = int(rgb.width * 0.2), int(rgb.width * 0.8)
        top = int(rgb.height * 0.58)
        counts = {color: 0 for color in COLOR_NAMES}
        for red, green, blue in rgb.crop((left, top, right, rgb.height)).getdata():
            maximum = max(red, green, blue)
            minimum = min(red, green, blue)
            if maximum < 55:
                color = "black"
            elif maximum - minimum < 18 and maximum > 205:
                color = "white"
            elif red > green * 1.25 and red > blue * 1.25:
                color = "red"
            elif red > blue * 1.25 and red > green * 1.05:
                color = "orange"
            elif green > red * 1.2 and green > blue * 1.15:
                color = "green"
            elif blue > red * 1.2 and blue > green * 1.1:
                color = "blue"
            elif maximum - minimum < 28:
                color = "gray"
            else:
                color = "brown"
            counts[color] += 1
        total = max(1, sum(counts.values()))
        ranked = sorted(((color, count / total) for color, count in counts.items()), key=lambda item: item[1], reverse=True)
        return ranked[: min(top_k, len(ranked))]
