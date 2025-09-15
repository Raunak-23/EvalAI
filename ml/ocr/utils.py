# utils.py
import os
import random
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from collections import Counter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def cer(pred, truth):
    # Character error rate (Levenshtein)
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, truth) / len(truth)

def build_charset(mapping_file, extra_chars=""):
    """
    Build charset from IAM mapping file.
    mapping_file: lines of "imgname\ttext"
    returns charset (string), char2idx dict (with blank at idx 0)
    """
    chars = Counter()
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            text = parts[1]
            chars.update(text)
    # sort by frequency (optional) but deterministic order
    sorted_chars = sorted(list(chars.keys()))
    for ch in extra_chars:
        if ch not in sorted_chars:
            sorted_chars.append(ch)
    # we reserve idx 0 for CTC blank
    idx2char = ["<BLANK>"] + sorted_chars
    char2idx = {c: i+1 for i, c in enumerate(sorted_chars)}
    char2idx["<BLANK>"] = 0
    return "".join(sorted_chars), char2idx, idx2char
