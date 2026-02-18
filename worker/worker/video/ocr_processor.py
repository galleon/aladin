"""
Multi-scale PaddleOCR pipeline for video frames.

Inspired by scene-text OCR for motion-blurred images: multi-scale passes,
CLAHE + unsharp preprocessing, per-detection confidence filtering,
IoU-based deduplication, and text normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import cv2
import numpy as np


@dataclass
class OcrHit:
    text: str
    conf: float
    # shape (4,2) float32 in ORIGINAL image coords
    # Note: not frozen — np.ndarray fields cannot be truly immutable in a dataclass.
    box: np.ndarray
    scale_tag: str


def ensure_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Failed to load image.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clahe_l_channel(bgr: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def unsharp_mask(bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
    return sharp


def preprocess_for_ocr(
    bgr: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.5,
    unsharp_sigma: float = 1.1,
    unsharp_amount: float = 0.9,
) -> np.ndarray:
    x = bgr
    if use_clahe:
        x = clahe_l_channel(x, clip_limit=clahe_clip_limit)
    if unsharp_amount > 0 and unsharp_sigma > 0:
        x = unsharp_mask(x, sigma=unsharp_sigma, amount=unsharp_amount)
    return x


def box_to_aabb(box: np.ndarray) -> Tuple[float, float, float, float]:
    xs = box[:, 0]
    ys = box[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou_aabb(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def normalize_text(s: str) -> str:
    s2 = s.strip()
    s2 = " ".join(s2.split())
    return s2


def run_paddle_ocr(
    ocr: Any,
    bgr: np.ndarray,
    downscale_factor: float,
    scale_tag: str,
    min_conf: float,
    use_angle_cls: bool = True,
) -> List[OcrHit]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = ocr.ocr(rgb, cls=use_angle_cls)
    hits: List[OcrHit] = []
    if not out or out[0] is None:
        return hits

    for item in out[0]:
        box_pts = np.array(item[0], dtype=np.float32)
        text, conf = item[1]
        if text is None:
            continue
        text_n = normalize_text(str(text))
        if not text_n:
            continue
        conf_f = float(conf) if conf is not None else 0.0
        if conf_f < min_conf:
            continue
        # downscale_factor < 1.0 when image was shrunk; divide to get original coords
        box_orig = box_pts / downscale_factor
        hits.append(OcrHit(text=text_n, conf=conf_f, box=box_orig, scale_tag=scale_tag))
    return hits


def merge_hits(hits: List[OcrHit], merge_iou: float) -> List[OcrHit]:
    if not hits:
        return []
    hits_sorted = sorted(hits, key=lambda h: h.conf, reverse=True)
    kept: List[OcrHit] = []
    kept_aabbs: List[Tuple[float, float, float, float]] = []

    for h in hits_sorted:
        aabb_h = box_to_aabb(h.box)
        suppress = False
        for i, k in enumerate(kept):
            aabb_k = kept_aabbs[i]
            iou = iou_aabb(aabb_h, aabb_k)
            if iou >= merge_iou:
                suppress = True
                break
            if h.text.lower() == k.text.lower() and iou >= (merge_iou * 0.6):
                suppress = True
                break
        if not suppress:
            kept.append(h)
            kept_aabbs.append(aabb_h)
    return kept


def ocr_single_image(
    bgr: np.ndarray,
    ocr: Any,
    max_sides: List[int],
    min_conf: float = 0.55,
    merge_iou: float = 0.35,
    keep_topk: int = 200,
    use_angle_cls: bool = True,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.5,
    unsharp_sigma: float = 1.1,
    unsharp_amount: float = 0.9,
) -> str:
    """
    Run multi-scale OCR on a single BGR image. Returns joined text lines.
    """
    try:
        bgr = ensure_uint8_bgr(bgr)
    except ValueError:
        return ""

    all_hits: List[OcrHit] = []
    seen_factors: set = set()
    for ms in max_sides:
        resized, factor = resize_max_side(bgr, ms)
        # Skip if this downscale factor was already processed (e.g. image smaller than max_side)
        if factor in seen_factors:
            continue
        seen_factors.add(factor)
        pre = preprocess_for_ocr(
            resized,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )
        tag = f"max_side={ms}"
        hits = run_paddle_ocr(
            ocr=ocr,
            bgr=pre,
            downscale_factor=factor,
            scale_tag=tag,
            min_conf=min_conf,
            use_angle_cls=use_angle_cls,
        )
        all_hits.extend(hits)

    merged = merge_hits(all_hits, merge_iou=merge_iou)
    # Cap by confidence, then sort spatially (top→bottom, left→right) for LLM readability
    merged = sorted(merged, key=lambda h: h.conf, reverse=True)[:keep_topk]
    merged = sorted(merged, key=lambda h: (h.box.mean(axis=0)[1], h.box.mean(axis=0)[0]))
    return " ".join(h.text for h in merged).strip()
