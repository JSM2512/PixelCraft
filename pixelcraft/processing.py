import io
import tempfile
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    img = np.array(image)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def adaptive_thresholding(image: np.ndarray) -> np.ndarray:
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    M, N = 5, 5
    g = image

    padded_image = np.pad(g, ((N // 2, N // 2), (M // 2, M // 2)), "reflect")

    lVar = np.zeros(g.shape)
    lM = np.zeros(g.shape)

    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            window = padded_image[i : i + M, j : j + N]
            lM[i, j] = np.mean(window)
            lVar[i, j] = np.var(window)

    nVar = np.sum(lVar) / (g.shape[0] * g.shape[1])
    lVar = np.maximum(lVar, nVar)
    ratio = nVar / lVar

    adaptive_filtered_image = g - ratio * (g - lM)
    return adaptive_filtered_image


def canny_edge_detection(image: Image.Image) -> np.ndarray:
    image = np.array(image)

    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)
    return edges


def normalize_for_streamlit(img: np.ndarray) -> np.ndarray:
    """
    Make sure image is 3-channel and float [0..1] or uint8 [0..255] acceptable.
    We keep your behavior: normalize to [0..1] when values > 1.
    """
    if len(img.shape) == 2:  # grayscale
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    if img.max() > 1:
        img = img / 255.0

    return img


def save_image_to_tempfile(image_bgr_or_rgb: np.ndarray) -> str:
    """
    Saves an image to a temporary .png file and returns its path.
    Input is assumed BGR (from OpenCV) for color images.
    """
    image = image_bgr_or_rgb

    # If float in [0..1], convert to uint8
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # If image is grayscale but in 3 channels already, cvtColor still works only if 3 channel.
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Convert BGR->RGB for PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_pil.save(tmp_file, format="PNG")
        return tmp_file.name


def file_bytes_for_download(tmp_file_path: str) -> bytes:
    with open(tmp_file_path, "rb") as f:
        return f.read()