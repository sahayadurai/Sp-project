"""Dataset loading and caption/image normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from datasets import load_dataset


PARQUET_API = "https://datasets-server.huggingface.co/parquet"


def load_flickr30k(split: str = "train", cache_dir: str | None = None, streaming: bool = True):
    """Load Flickr30k without invoking its retired legacy dataset script.

    The Hub's converted data has one physical ``TEST/test`` split, while each
    row contains the original ``train``, ``val``, or ``test`` split name.
    """
    query = urlencode({"dataset": "nlphuji/flickr30k"})
    with urlopen(f"{PARQUET_API}?{query}") as response:
        files = response.read()
    parquet_files = [item["url"] for item in json.loads(files)["parquet_files"]]
    dataset = load_dataset(
        "parquet",
        data_files={"test": parquet_files},
        split="test",
        cache_dir=cache_dir,
        streaming=streaming,
    )
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    return dataset.filter(lambda row: row.get("split") == split)


def image_id(record: dict[str, Any], index: int) -> str:
    for key in ("image_id", "img_id", "filename", "image"):
        value = record.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return str(index)


def captions(record: dict[str, Any]) -> list[str]:
    value = record.get("caption", record.get("captions", []))
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = value.get("text", [])
    result = []
    for item in value or []:
        result.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
    return [item for item in result if item]


def image_from_record(record: dict[str, Any]):
    value = record.get("image")
    if value is None:
        raise ValueError("The dataset record does not contain an image column.")
    if hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes"):
        from io import BytesIO
        from PIL import Image
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    path = Path(str(value))
    if path.exists():
        from PIL import Image
        return Image.open(path).convert("RGB")
    raise ValueError("Unsupported image value in dataset record.")
