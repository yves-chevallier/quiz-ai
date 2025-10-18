"""
Utility helpers around the OpenAI client used in the CLI commands.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

DEFAULT_VISION_MODEL = "gpt-5"


@dataclass(frozen=True)
class VisionRequest:
    """Description of a single vision prompt request."""

    prompt: str
    image_path: Path
    model: str = DEFAULT_VISION_MODEL


def build_openai_client(timeout: float = 300.0, max_connections: int = 10) -> OpenAI:
    """
    Instantiate an OpenAI client with an HTTPX transport configured for long-running calls.
    """
    http_client = httpx.Client(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
    )
    return OpenAI(http_client=http_client)


def image_file_to_data_url(image_path: Path) -> str:
    """
    Convert an image file to a data URL suitable for the OpenAI vision API.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/webp" if suffix == ".webp" else "image/jpeg"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def call_vision(
    client: OpenAI,
    *,
    prompt: str,
    image_data_url: str,
    model: str = DEFAULT_VISION_MODEL,
    user: Optional[str] = None,
) -> Any:
    """
    Execute a vision request and return the raw OpenAI response object.
    """
    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        user=user,
    )
