from pathlib import Path
import json
import io
import base64
from typing import Any, List
from PIL import Image
from pdf2image import convert_from_path

from sandbox.pipeline import OUT_DIR

# ---------- OUTILS FICHIERS ----------
def ensure_out_dir(path: Path = OUT_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pil_image_to_jpeg_data_url(
    img: Image.Image, max_side: int = 1600, quality: int = 65
) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------- OUTILS IMAGES ----------
def pil_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pdf_to_pil_images_in_memory(
    pdf_path: Path, dpi: int = DPI_IMAGES
) -> List[Image.Image]:
    return convert_from_path(str(pdf_path), dpi=dpi)
