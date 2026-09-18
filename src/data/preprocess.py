import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


def deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def horizontal_projection(binary: np.ndarray) -> np.ndarray:
    return np.sum(binary, axis=1)


def segment_lines(binary: np.ndarray, min_line_height: int = 15, merge_gap: int = 8) -> list[tuple[int, int]]:
    projection = horizontal_projection(binary)
    threshold = np.max(projection) * 0.05 if np.max(projection) > 0 else 0
    is_line = projection > threshold

    lines = []
    start = None
    for i, val in enumerate(is_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_line_height:
                lines.append((start, i))
            start = None
    if start is not None and len(is_line) - start >= min_line_height:
        lines.append((start, len(is_line)))

    if not lines:
        return []

    merged = [lines[0]]
    for s, e in lines[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def crop_lines(image: np.ndarray, lines: list[tuple[int, int]], padding: int = 5) -> list[np.ndarray]:
    h, w = image.shape[:2]
    crops = []
    for y_start, y_end in lines:
        s = max(0, y_start - padding)
        e = min(h, y_end + padding)
        crop = image[s:e, :]
        if crop.size > 0:
            crops.append(crop)
    return crops


def is_blank_line(line_image: np.ndarray, threshold: float = 0.02) -> bool:
    white_ratio = np.sum(line_image > 128) / line_image.size
    return white_ratio > (1 - threshold)


def split_line_to_patches(
    line_image: np.ndarray,
    target_height: int = 64,
    patch_width: int = 384,
    min_content_ratio: float = 0.05,
) -> list[np.ndarray]:
    h, w = line_image.shape[:2]

    scale = target_height / h
    new_w = max(int(w * scale), patch_width)
    resized = cv2.resize(line_image, (new_w, target_height), interpolation=cv2.INTER_AREA)

    if new_w <= patch_width:
        pad = np.zeros((target_height, patch_width - new_w), dtype=resized.dtype)
        patch = np.hstack([resized, pad])
        content_ratio = np.sum(patch > 128) / patch.size
        if content_ratio >= min_content_ratio:
            return [patch]
        return []

    step = patch_width // 2
    patches = []
    for x in range(0, new_w - patch_width + 1, step):
        patch = resized[:, x : x + patch_width]
        content_ratio = np.sum(patch > 128) / patch.size
        if content_ratio >= min_content_ratio:
            patches.append(patch)

    if not patches and new_w > 0:
        patch = resized[:, :patch_width]
        if patch.shape[1] < patch_width:
            pad = np.zeros((target_height, patch_width - patch.shape[1]), dtype=patch.dtype)
            patch = np.hstack([patch, pad])
        patches.append(patch)

    return patches


def preprocess_page(
    image_path: Path,
    output_dir: Path,
    person_prefix: str = "patch",
    target_height: int = 64,
    patch_width: int = 384,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path, grayscale=True)
    binary = binarize(image, method="otsu")
    binary = deskew(binary)
    lines = segment_lines(binary)
    crops = crop_lines(binary, lines)

    saved = []
    idx = 0
    for line_img in crops:
        if is_blank_line(line_img):
            continue
        patches = split_line_to_patches(line_img, target_height, patch_width)
        for patch in patches:
            out_path = output_dir / f"{person_prefix}_{idx:04d}.png"
            cv2.imwrite(str(out_path), patch)
            saved.append(out_path)
            idx += 1
    return saved


def prepare_person_data(
    raw_person_dir: Path,
    patches_person_dir: Path,
    person_prefix: str = "patch",
    target_height: int = 64,
    patch_width: int = 384,
) -> list[Path]:
    all_patches = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = sorted(f for f in raw_person_dir.iterdir() if f.suffix.lower() in image_extensions)

    for img_file in files:
        patches = preprocess_page(
            img_file,
            patches_person_dir,
            person_prefix=person_prefix,
            target_height=target_height,
            patch_width=patch_width,
        )
        all_patches.extend(patches)
    return all_patches
