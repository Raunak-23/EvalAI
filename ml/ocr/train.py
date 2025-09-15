# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from config import *
from dataset import IAMWordDataset, collate_fn
from model import CRNNWithAttention
from utils import set_seed, make_dirs, build_charset, cer

set_seed(SEED)
make_dirs(MODEL_DIR, LOG_DIR)

def compute_input_lengths(img_widths, cnn_module, H=IMG_HEIGHT):
    """
    Approximate time-steps (T) after CNN for each image width.
    We can pass a dummy tensor of size (1, C, H, W) through CNN to get W' (T).
    """
    device = next(cnn_module.parameters()).device
    dummy = torch.zeros((1, NUM_CHANNELS, H, int(img_widths.max().item())), device=device)
    with torch.no_grad():
        feats = cnn_module(dummy)
        _, _, Hc, Wc = feats.shape
    return Wc

def train():
    # Build charset
    _, char2idx, idx2char = build_charset(IAM_MAPPING)
    num_classes = len(idx2char)
    print("Num classes (including no-blank index reserved):", num_classes)
    # Update config variable if needed
    # dataset
    ds = IAMWordDataset(IAM_MAPPING, img_root=DATA_ROOT, charset=char2idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_ds = IAMWordDataset(IAM_MAPPING, img_root=DATA_ROOT, charset=char2idx)  # ideally separate val split
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = CRNNWithAttention(num_classes=num_classes, in_channels=NUM_CHANNELS).to(DEVICE)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        for imgs, labels_concat, label_lengths, widths, texts, names in pbar:
            imgs = imgs.to(DEVICE)
            labels_concat = labels_concat.to(DEVICE)
            # forward
            log_probs = model(imgs)  # [B, T, C]
            B, T, C = log_probs.size()
            log_probs_perm = log_probs.permute(1, 0, 2)  # [T, B, C]
            # input lengths: computed from widths via a dummy pass or approximated by CNN stride
            with torch.no_grad():
                # an approximation: pass widths.max() through cnn to get T_max
                input_T = compute_input_lengths(widths, model.cnn, H=IMG_HEIGHT)
            input_lengths = torch.full((B,), fill_value=input_T, dtype=torch.long)
            # prepare label lengths and labels
            # labels_concat is concatenation of labels in batch
            loss = ctc_loss(log_probs_perm, labels_concat, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(loader)
        # validation (simple)
        model.eval()
        total_cer = 0.0
        n = 0
        with torch.no_grad():
            for imgs, labels_concat, label_lengths, widths, texts, names in val_loader:
                imgs = imgs.to(DEVICE)
                log_probs = model(imgs)  # [B, T, C]
                # greedy decode
                preds = greedy_decode_batch(log_probs, idx2char)
                for p, t in zip(preds, texts):
                    total_cer += cer(p, t)
                    n += 1
        avg_cer = total_cer / max(1, n)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} val CER: {avg_cer:.4f}")
        scheduler.step(avg_loss)
        # save
        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "charlist": idx2char
            }, os.path.join(MODEL_DIR, f"crnn_epoch_{epoch}.pth"))

def greedy_decode_batch(log_probs_batch, idx2char):
    """
    log_probs_batch: [B, T, C] (log softmax)
    returns list of decoded strings
    """
    with torch.no_grad():
        probs = log_probs_batch.exp()  # [B, T, C]
        preds = torch.argmax(probs, dim=-1).cpu().numpy()  # [B, T]
    results = []
    for seq in preds:
        last = None
        out = []
        for idx in seq:
            if idx != last and idx != 0:  # skip blanks and repeats
                out.append(idx2char[idx])
            last = idx
        results.append("".join(out))
    return results

if __name__ == "__main__":
    train()
