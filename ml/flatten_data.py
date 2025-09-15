#!/usr/bin/env python3
"""
flatten_iam_words.py
Flatten the IAM-word nested folder structure.
Before:  iam_words/words/a01/a01-000u/a01-000u-00-00.png
After:   iam_words/words_flat/a01-000u-00-00.png
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm   # pip install tqdm  (optional, nicer progress bar)

# ------------- CONFIG --------------
SRC = Path("D:/EvalAI/data/handwritten/data")   # change if necessary
DST = Path("D:/EvalAI/data/handwritten/data_flat") # same parent dir
# -----------------------------------

DST.mkdir(exist_ok=True)

pngs = list(SRC.rglob("*.png"))
print(f"Found {len(pngs)} PNG files.")

for file in tqdm(pngs, desc="Copying"):
    target = DST / file.name
    # handle extremely unlikely name collision
    counter = 1
    stem = target.stem
    suffix = target.suffix
    while target.exists():
        target = DST / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(file, target)

print("Done – flat folder:", DST.resolve())