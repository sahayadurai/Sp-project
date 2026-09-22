"""Create lightweight, reproducible VQA supervision from Flickr30k captions."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QAExample:
    image_id: str
    caption: str
    question: str
    answer: str


STOPWORDS = {
    "a", "an", "the", "and", "with", "in", "on", "at", "of", "to", "for",
    "near", "over", "under", "by", "is", "are", "this", "that", "from",
}
COLORS = {"black", "white", "red", "blue", "green", "yellow", "orange", "brown", "pink", "gray", "grey"}


def _tokens(caption: str) -> list[str]:
    return re.findall(r"[a-z]+", caption.lower())


def _noun_candidates(caption: str) -> list[str]:
    words = [word for word in _tokens(caption) if word not in STOPWORDS and len(word) > 2]
    return list(dict.fromkeys(words))


def caption_to_examples(image_id: str, caption: str) -> list[QAExample]:
    """Generate answerable lexical, color, and count questions from one caption."""
    words = _tokens(caption)
    examples: list[QAExample] = []
    nouns = _noun_candidates(caption)

    if nouns:
        examples.append(QAExample(image_id, caption, "What is mentioned in the image?", nouns[0]))
        examples.append(QAExample(image_id, caption, "What is the main object?", nouns[0]))
    for color in COLORS.intersection(words):
        examples.append(QAExample(image_id, caption, "What color is visible?", color))
    counts = Counter(words)
    repeated = [(word, count) for word, count in counts.items() if count > 1 and word not in STOPWORDS]
    if repeated:
        word, count = max(repeated, key=lambda item: item[1])
        examples.append(QAExample(image_id, caption, "How many times is the object mentioned?", str(count)))
        examples.append(QAExample(image_id, caption, "What object appears repeatedly?", word))

    # These templates provide useful negative supervision without requiring a parser.
    if any(word in words for word in ("man", "boy", "person", "people", "woman", "girl")):
        examples.append(QAExample(image_id, caption, "Is there a person?", "yes"))
    if any(word in words for word in ("dog", "cat", "horse", "bird", "animal")):
        examples.append(QAExample(image_id, caption, "Is there an animal?", "yes"))
    return examples


def build_examples(records: Iterable[dict], max_captions: int | None = None) -> list[QAExample]:
    examples: list[QAExample] = []
    for index, record in enumerate(records):
        if max_captions is not None and index >= max_captions:
            break
        image_id = str(record.get("image_id", record.get("img_id", index)))
        captions = record.get("caption", record.get("captions", ""))
        if isinstance(captions, str):
            captions = [captions]
        for caption in captions or []:
            if isinstance(caption, dict):
                caption = caption.get("text", "")
            if caption:
                examples.extend(caption_to_examples(image_id, str(caption)))
    return examples
