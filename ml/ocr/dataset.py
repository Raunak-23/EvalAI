# dataset.py
import os
import math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_HEIGHT, IMG_MAX_WIDTH, MEAN, STD, BLANK_IDX
from utils import build_charset

class IAMWordDataset(Dataset):
    """
    Simple dataset for IAM mapping file where each line contains: image_path \t transcription
    Image path can be absolute or relative to a root passed in construct.
    """
    def __init__(self, mapping_file, img_root="", charset=None, transform=None, max_width=IMG_MAX_WIDTH, img_height=IMG_HEIGHT):
        super().__init__()
        self.img_root = img_root
        self.entries = []
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split("\t")
                if len(parts)<2: continue
                img, text = parts[0], parts[1]
                self.entries.append((img, text))
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.max_width = max_width
        self.img_height = img_height
        if charset is None:
            # build charset from mapping file
            _, self.char2idx, self.idx2char = build_charset(mapping_file)
        else:
            self.char2idx = charset
            self.idx2char = [k for k, v in sorted(charset.items(), key=lambda kv: kv[1])]
        self.blank_idx = BLANK_IDX

    def __len__(self):
        return len(self.entries)

    def _load_image(self, img_path):
        p = img_path
        if not os.path.isabs(p):
            p = os.path.join(self.img_root, img_path)
        im = Image.open(p).convert("RGB")
        # resize preserving aspect ratio to height = img_height
        w, h = im.size
        new_h = self.img_height
        new_w = int(w * (new_h / h))
        if new_w > self.max_width:
            new_w = self.max_width
            # rescale height to preserve ratio
            new_h = int(h * (new_w / w))
            im = im.resize((new_w, new_h), Image.BILINEAR)
            # then pad/resize to img_height (we will pad vertically)
            if new_h != self.img_height:
                # pad/tile center vertical
                pad_v = self.img_height - new_h
                new_im = Image.new("RGB", (new_w, self.img_height), (255,255,255))
                new_im.paste(im, (0, pad_v//2))
                im = new_im
        else:
            im = im.resize((new_w, new_h), Image.BILINEAR)
        return im

    def __getitem__(self, idx):
        img_name, text = self.entries[idx]
        img = self._load_image(img_name)
        img = self.transform(img)
        # convert text to indices
        label = [self.char2idx.get(ch, self.blank_idx) for ch in text]
        label = torch.tensor(label, dtype=torch.long)
        return img, label, text, img_name

def collate_fn(batch):
    """
    Pads the batch images to same width (max width in batch) and pads labels.
    Returns: images tensor [B, C, H, W], labels padded, label_lengths, input_widths (for CTC)
    """
    imgs, labels, texts, names = zip(*batch)
    bs = len(imgs)
    C, H = imgs[0].shape[0], imgs[0].shape[1]
    widths = [im.shape[2] for im in imgs]
    max_w = max(widths)
    padded = torch.zeros((bs, C, H, max_w), dtype=imgs[0].dtype)
    for i, im in enumerate(imgs):
        w = im.shape[2]
        padded[i, :, :, :w] = im

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels) if len(labels) > 0 else torch.tensor([], dtype=torch.long)
    return padded, labels_concat, label_lengths, torch.tensor(widths, dtype=torch.long), texts, names
