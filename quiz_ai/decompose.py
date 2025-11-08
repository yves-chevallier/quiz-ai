"""
Utilities to render a PDF into page images and crop regions defined
in an Anchors schema (see `.anchors` module).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Iterable

import fitz  # PyMuPDF
from PIL import Image

# Import the Pydantic models produced by your anchors module
# (names assumed from your previous code)
from .anchors import Anchors, PageAnchors, BoxRegion  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class PageImage:
    """Represents a rendered page image on disk."""

    page_index: int
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CropBox:
    """Pillow box coordinates (left, top, right, bottom) in pixels."""

    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """
        Return the crop box coordinates as a tuple.
        """
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class RegionCrop:
    """Represents a cropped region for a specific question/part/subpart."""

    question_id: int
    page_index: int
    region_index: int
    path: Path
    part: Optional[int] = None
    subpart: Optional[int] = None


@dataclass(frozen=True)
class BoxCrop:
    """Represents a cropped named box."""

    name: str
    page_index: int
    path: Path


def mm_region_to_pixel_box(
    img_w: int,
    img_h: int,
    page_height_mm: float,
    y_start_mm: float,
    y_end_mm: float,
) -> CropBox:
    """
    Convert a region [y_start_mm, y_end_mm] into a Pillow crop box.
    JSON/PDF space: y=0 at bottom; image space: y=0 at top.

    Ensures at least 1px height and clamps to image bounds.
    """
    if page_height_mm <= 0:
        raise ValueError("page_height_mm must be > 0")

    scale = img_h / page_height_mm  # vertical px/mm

    top_px = int(round(img_h - (y_end_mm * scale)))
    bottom_px = int(round(img_h - (y_start_mm * scale)))

    top_px = max(0, min(img_h, top_px))
    bottom_px = max(0, min(img_h, bottom_px))
    if bottom_px <= top_px:
        bottom_px = min(img_h, top_px + 1)

    return CropBox(left=0, top=top_px, right=img_w, bottom=bottom_px)


def mm_box_to_pixel_box(
    box: BoxRegion,
    img_w: int,
    img_h: int,
    page_width_mm: float,
    page_height_mm: float,
    margin_mm: float = 0.0,
) -> CropBox:
    """
    Convert a rectangular box defined in mm to pixel coordinates, applying an optional margin.
    """
    if page_width_mm <= 0 or page_height_mm <= 0:
        raise ValueError("page dimensions must be > 0")

    scale_x = img_w / page_width_mm
    scale_y = img_h / page_height_mm

    x_min = max(0.0, box.x_min_mm - margin_mm)
    x_max = min(page_width_mm, box.x_max_mm + margin_mm)
    y_min = max(0.0, box.y_min_mm - margin_mm)
    y_max = min(page_height_mm, box.y_max_mm + margin_mm)

    left_px = int(round(x_min * scale_x))
    right_px = int(round(x_max * scale_x))
    top_px = int(round(img_h - (y_max * scale_y)))
    bottom_px = int(round(img_h - (y_min * scale_y)))

    left_px = max(0, min(img_w, left_px))
    right_px = max(0, min(img_w, right_px))
    top_px = max(0, min(img_h, top_px))
    bottom_px = max(0, min(img_h, bottom_px))

    if right_px <= left_px:
        right_px = min(img_w, left_px + 1)
    if bottom_px <= top_px:
        bottom_px = min(img_h, top_px + 1)

    return CropBox(left=left_px, top=top_px, right=right_px, bottom=bottom_px)


def _save_pillow_image(img: Image.Image, dest: Path, quality: int = 95) -> None:
    """
    Save a Pillow image with sane defaults by format.
    """

    img.save(dest, format="JPEG", quality=quality, optimize=True)


def _pixmap_to_pillow(pix: fitz.Pixmap) -> Image.Image:
    """
    Convert a PyMuPDF pixmap to a Pillow RGB image.
    """
    # Force RGB without alpha; callers disable alpha in get_pixmap already.
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


class PdfCutter:
    """
    Service that renders a PDF into images and crops regions provided by an Anchors schema.

    Parameters
    ----------
    dpi : int
        Rendering DPI for PDF pages (72 * scale).
    img_format : str
        Output image format for rendered pages ('jpg', 'png', 'webp').
    quality : int
        Quality for JPEG/WEBP.
    """

    def __init__(self, dpi: int = 150, quality: int = 95) -> None:
        if not 30 <= dpi <= 1200:
            raise ValueError("dpi must be in [30, 1200]")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be in [1, 100]")

        self.dpi = dpi
        self.quality = quality

    def render_pdf_to_images(self, pdf_path: Path, out_dir: Path) -> List[PageImage]:
        """
        Render each page of a PDF to an image file in `out_dir`.

        Returns a list of PageImage with page index, path, and size.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[PageImage] = []

        with fitz.open(pdf_path) as doc:
            # pylint: disable=consider-using-enumerate
            page_count = doc.page_count
            scale = self.dpi / 72.0
            mat = fitz.Matrix(scale, scale)

            for pno in range(page_count):
                page = doc[pno]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = _pixmap_to_pillow(pix)

                out_path = out_dir / f"{pdf_path.stem}_page{pno+1}.jpg"
                _save_pillow_image(img, out_path, self.quality)

                written.append(
                    PageImage(
                        page_index=pno,
                        path=out_path,
                        width=img.width,
                        height=img.height,
                    )
                )

        return written

    def crop_page_image(
        self,
        image_path: Path,
        regions_mm: Iterable[Dict[str, float]],
        page_height_mm: float,
        output_dir: Path,
        page_index: int,
        base_output_stem: Optional[str] = None,
        out_ext: Optional[str] = None,
    ) -> List[RegionCrop]:
        """
        Open a page image and crop all provided regions. Returns the list of created files.
        """
        written: List[RegionCrop] = []
        if not image_path.exists():
            return written

        output_dir.mkdir(parents=True, exist_ok=True)
        base_stem = base_output_stem or image_path.stem
        ext = (out_ext or image_path.suffix.lstrip(".")).lower()

        label_counts: Dict[str, int] = {}

        with Image.open(image_path) as im:
            if ext in {"jpg", "jpeg"} and im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size

            for i, reg in enumerate(regions_mm, start=1):
                y_start = float(reg["y_start_mm"])
                y_end = float(reg["y_end_mm"])
                qnum = int(reg["qnum"])
                part_val = reg.get("part")
                subpart_val = reg.get("subpart")
                part_idx = int(part_val) if isinstance(part_val, (int, float)) else None
                subpart_idx = int(subpart_val) if isinstance(subpart_val, (int, float)) else None
                box = mm_region_to_pixel_box(w, h, page_height_mm, y_start, y_end)

                cropped = im.crop(box.as_tuple())
                label_parts = [str(qnum)]
                if part_idx:
                    label_parts.append(str(part_idx))
                if subpart_idx:
                    label_parts.append(str(subpart_idx))
                label = "_".join(label_parts)
                if not label:
                    label = f"{qnum}"
                count = label_counts.get(label, 0)
                label_counts[label] = count + 1
                file_label = f"{label}_{count+1}" if count else label
                out_name = f"{base_stem}_{file_label}.{ext}"
                out_path = output_dir / out_name
                _save_pillow_image(cropped, out_path, self.quality)
                written.append(
                    RegionCrop(
                        question_id=qnum,
                        page_index=page_index,
                        region_index=i,
                        path=out_path,
                        part=part_idx,
                        subpart=subpart_idx,
                    )
                )

        return written

    def crop_regions_for_pdf(
        self,
        pdf_path: Path,
        anchors: Anchors,
        out_dir: Path,
        skip_empty: bool = False,
    ) -> List[RegionCrop]:
        """
        Render the PDF to images and crop each page according to `anchors.pages[*].regions_mm`.

        Returns a flat list of all cropped file paths.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        page_images = self.render_pdf_to_images(pdf_path, out_dir)

        pages_by_index: Dict[int, PageAnchors] = {p.page_index: p for p in anchors.pages}
        all_written: List[RegionCrop] = []

        for page_img in page_images:
            entry = pages_by_index.get(page_img.page_index)
            if entry is None:
                if skip_empty:
                    continue
                # No entry -> nothing to crop
                continue

            regions_mm = [r.model_dump() for r in entry.regions_mm]  # Pydantic v2 -> dicts
            if skip_empty and not regions_mm:
                continue

            page_height_mm = float(entry.page_height_mm or 0.0)
            if page_height_mm <= 0:
                continue

            written = self.crop_page_image(
                image_path=page_img.path,
                regions_mm=regions_mm,
                page_height_mm=page_height_mm,
                output_dir=out_dir,
                page_index=page_img.page_index,
                base_output_stem=page_img.path.stem,
            )
            all_written.extend(written)

        return all_written

    def crop_box_images(
        self,
        pdf_path: Path,
        anchors: Anchors,
        out_dir: Path,
        margin_mm: float = 0.0,
        box_names: Optional[Iterable[str]] = None,
    ) -> List[BoxCrop]:
        """
        Render the PDF and crop each named box (e.g., cover name field) with an optional margin.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = out_dir / "_pages"
        page_images = self.render_pdf_to_images(pdf_path, pages_dir)
        page_images_map = {img.page_index: img for img in page_images}
        requested = {name for name in box_names} if box_names else None

        written: List[BoxCrop] = []
        for page_entry in anchors.pages:
            if not page_entry.boxes_mm:
                continue
            page_image = page_images_map.get(page_entry.page_index)
            if not page_image:
                continue
            with Image.open(page_image.path) as im:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                w, h = im.size
                for box in page_entry.boxes_mm:
                    if requested and box.name not in requested:
                        continue
                    crop_box = mm_box_to_pixel_box(
                        box=box,
                        img_w=w,
                        img_h=h,
                        page_width_mm=float(page_entry.page_width_mm),
                        page_height_mm=float(page_entry.page_height_mm),
                        margin_mm=margin_mm,
                    )
                    cropped = im.crop(crop_box.as_tuple())
                    out_name = f"page{page_entry.page_index+1}_{box.name}.jpg"
                    out_path = out_dir / out_name
                    _save_pillow_image(cropped, out_path, self.quality)
                    written.append(
                        BoxCrop(
                            name=box.name,
                            page_index=page_entry.page_index,
                            path=out_path,
                        )
                    )
        return written


# Convenience functional API, if you prefer functions over the class


def render_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 150,
    quality: int = 95,
) -> List[PageImage]:
    """
    Functional wrapper around PdfCutter.render_pdf_to_images.
    """
    return PdfCutter(dpi=dpi, quality=quality).render_pdf_to_images(pdf_path, out_dir)


def crop_regions_for_pdf(
    pdf_path: Path,
    anchors: Anchors,
    out_dir: Path,
    dpi: int = 150,
    quality: int = 95,
    skip_empty: bool = False,
) -> List[RegionCrop]:
    """
    Functional wrapper that renders the PDF then crops according
    to anchors, returning all cropped files.
    """
    cutter = PdfCutter(dpi=dpi, quality=quality)
    return cutter.crop_regions_for_pdf(
        pdf_path=pdf_path,
        anchors=anchors,
        out_dir=out_dir,
        skip_empty=skip_empty,
    )
