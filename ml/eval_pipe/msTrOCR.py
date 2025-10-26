import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2, numpy as np

# Load model (takes ~2 GB RAM)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model.eval()

def preprocess_image(pil_img):
    # Convert to grayscale, deskew, binarize
    img = np.array(pil_img) 
    # Deskew using OpenCV (optional)
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Binarize
    _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def ocr_handwritten(pil_image):
    pil_image = preprocess_image(pil_image)
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    # If you have a GPU, move the tensor: pixel_values = pixel_values.to('cuda')
    generated_ids = ocr_model.generate(pixel_values, max_length=256)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# Example
img = Image.open(r"D:\EvalAI\data\test_data\eng_AF_004.jpg")
print(ocr_handwritten(img))