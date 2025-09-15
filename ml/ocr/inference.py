# inference.py
import torch
from model import CRNNWithAttention
from dataset import IAMWordDataset, collate_fn
from utils import build_charset
from config import DEVICE, NUM_CHANNELS, IMG_HEIGHT
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def load_model(path, num_classes, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = CRNNWithAttention(num_classes=num_classes, in_channels=NUM_CHANNELS)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model, ckpt.get("charlist", None)

def preprocess_image(pil_img, img_height=IMG_HEIGHT):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # resize preserving height
    w, h = pil_img.size
    new_w = int(w * (img_height / h))
    pil_img = pil_img.resize((new_w, img_height), Image.BILINEAR)
    return transform(pil_img).unsqueeze(0)  # [1, C, H, W]

def greedy_decode(log_probs, idx2char):
    probs = log_probs.exp()
    seq = torch.argmax(probs, dim=-1)[0].cpu().numpy()  # [T]
    out = []
    last = None
    for idx in seq:
        if idx != last and idx != 0:
            out.append(idx2char[idx])
        last = idx
    return "".join(out)

def predict_image(model, pil_img, idx2char):
    x = preprocess_image(pil_img)
    x = x.to(DEVICE)
    with torch.no_grad():
        log_probs = model(x)  # [B=1, T, C]
        pred = greedy_decode(log_probs, idx2char)
    return pred
