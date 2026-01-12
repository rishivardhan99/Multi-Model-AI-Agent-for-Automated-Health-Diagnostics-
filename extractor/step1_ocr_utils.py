# extractor/step1_ocr_utils.py
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Tesseract path (in Docker)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def preprocess_for_ocr(pil_img: Image.Image):
    """
    VERY FAST OCR PREPROCESSING:
    - Convert to grayscale
    - Slight blur (optional)
    - Otsu threshold (binarize)
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold → best for scanned lab reports
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(bw)

def ocr_image_to_text(pil_img: Image.Image, lang="eng", try_multiple=True):
    pre = preprocess_for_ocr(pil_img)

    # simple, fast configs
    if not try_multiple:
        return pytesseract.image_to_string(pre, lang=lang, config="--psm 6")

    configs = [
        "--psm 6",
        "--psm 3",
        "--psm 11",  # sparse text
        "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/.%()-<>,"
    ]

    best_text = ""
    best_score = 0

    def score(txt):
        return sum(ch.isdigit() for ch in txt)

    for cfg in configs:
        txt = pytesseract.image_to_string(pre, lang=lang, config=cfg)
        s = score(txt)
        if s > best_score:
            best_score = s
            best_text = txt

    return best_text
