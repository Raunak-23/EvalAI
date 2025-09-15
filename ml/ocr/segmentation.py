# segmentation.py
"""
Simple heuristic segmentation for pages -> lines -> words using OpenCV.
This is not as robust as an ML detector but works well for exam papers / IAM-like scanned pages.

Functions:
- pdf_to_images(pdf_path) -> list of PIL.Image pages (requires pdf2image)
- page_to_lines(image) -> list of PIL.Image line crops
- line_to_words(line_img) -> list of word crops (PIL.Image)
- heuristics to separate diagrams/equations: connected components with large area or non-text aspect ratio
"""
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, dpi=200):
    pages = convert_from_path(pdf_path, dpi=dpi)
    return pages

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def page_to_lines(pil_img, blur=5, morph_kernel=(50,5), min_line_height=10):
    cv = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # morphological closing to join chars in a line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # projection on vertical axis to find line bands
    horiz_proj = np.sum(closed, axis=1)
    # find splits
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(horiz_proj):
        if val > 0 and not in_line:
            in_line = True
            start = i
        elif val == 0 and in_line:
            end = i
            if end - start > min_line_height:
                crop = cv[start:end, :]
                lines.append(cv_to_pil(cv2.bitwise_not(crop)))
            in_line = False
    # post process: if none found, return whole page as line
    if not lines:
        lines = [pil_img]
    return lines

def line_to_words(pil_line_img, min_word_width=10):
    cv = pil_to_cv(pil_line_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # vertical projection to split words
    vert_proj = np.sum(th, axis=0)
    words = []
    in_word = False
    start = 0
    for j, val in enumerate(vert_proj):
        if val > 0 and not in_word:
            in_word = True
            start = j
        elif val == 0 and in_word:
            end = j
            if end - start > min_word_width:
                crop = cv[:, start:end]
                words.append(cv_to_pil(cv2.bitwise_not(crop)))
            in_word = False
    if not words:
        words = [pil_line_img]
    return words

def detect_diagrams_or_equations(pil_img, area_threshold_ratio=0.05):
    """
    Simple heuristic: if connected components occupy large areas or many non-text shapes, mark as diagram.
    """
    cv = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    h, w = th.shape
    total_area = h * w
    large_cc = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area / total_area > area_threshold_ratio:
            large_cc += 1
    return large_cc > 0
