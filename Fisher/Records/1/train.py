"""train_skullstrip_3dcnn.py ─ rev 2025-04-20
────────────────────────────────────────────
* **500-epoch ready**: logs per-epoch losses to TXT and PNG
* **Confusion matrix** after training (validation set)
* **Example slice montages** saved with actual series ID
* Progress bars via *tqdm*

Run:
    python train_skullstrip_3dcnn.py --epochs 500 --batch 2 [--regen_cache]
"""

# ── imports ─────────────────────────────────────────────────────────
from __future__ import annotations
import os, argparse, random, joblib, cv2, pydicom, SimpleITK as sitk, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.video import r3d_18
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── constants ───────────────────────────────────────────────────────
WINDOWS       = [(40, 80), (75, 215), (600, 2800)]  # WL/WW triplet
HU_RANGE      = (10, 50)                            # mask threshold
TARGET_SIZE   = 112                                 # slice H,W after resize
CACHE_FILE    = "skullstrip_dataset.pkl"
NUM_CLASSES   = 4
CLASS_NAMES   = ["1", "2", "3", "4"]

# ╭───────────────────────────────────────────────────────────────────╮
# │  Pre-processing helpers                                          │
# ╰───────────────────────────────────────────────────────────────────╯

def load_volume(series_dir: str) -> np.ndarray:
    files = [pydicom.dcmread(os.path.join(series_dir, f))
             for f in os.listdir(series_dir) if f.endswith(".dcm")]
    files.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([d.pixel_array for d in files]).astype(np.float32)
    return vol * files[0].RescaleSlope + files[0].RescaleIntercept


def skull_strip(vol_hu: np.ndarray) -> np.ndarray:
    img   = sitk.GetImageFromArray(vol_hu)
    mask  = sitk.BinaryThreshold(img, *HU_RANGE, 1, 0)
    cc    = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    brain   = sitk.BinaryThreshold(cc, largest, largest, 1, 0)
    return vol_hu * sitk.GetArrayFromImage(brain).astype(bool)


def apply_window(sl: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width/2, center + width/2
    np.clip(sl, lo, hi, out=sl)
    return sl


def preprocess_series(series_dir: str) -> torch.Tensor:
    vol = skull_strip(load_volume(series_dir))
    frames = []
    for sl in vol:
        sl = cv2.resize(sl, (TARGET_SIZE, TARGET_SIZE), cv2.INTER_AREA)
        chans = [ (apply_window(sl.copy(), *w) - (w[0] - w[1]/2)) / w[1]
                   for w in WINDOWS ]
        frames.append(np.stack(chans))
    arr = np.stack(frames, axis=1)          # [3,D,H,W]
    return torch.tensor(arr, dtype=torch.float32)

# ╭───────────────────────────────────────────────────────────────────╮
# │  Dataset + loader                                                │
# ╰───────────────────────────────────────────────────────────────────╯
class CTSeriesDataset(Dataset):
    """In-memory tensor list + label list + ID list."""
    def __init__(self, root: str, labels_csv: str, cache: str = CACHE_FILE, regen=False):
        import pandas as pd
        df = pd.read_csv(labels_csv)
        self.label_map = {sid: sc-1 for sid, sc in zip(df.series_id, df.score)}

        if os.path.exists(cache) and not regen:
            print(f"[INFO] loading cached tensors → {cache}")
            data = joblib.load(cache)
            if len(data) == 3:          # new format: vols, labels, ids
                self.vols, self.targets, self.ids = data
            else:                        # legacy 2‑tuple
                self.vols, self.targets = data
                self.ids = [f"legacy_{i}" for i in range(len(self.vols))]
                print("[WARN] legacy cache detected → ids filled with placeholders; run with --regen_cache later to upgrade.")
        else:
            self.vols, self.targets, self.ids = [], [], []
            print("[INFO] caching dataset (first time only)…")
            for sid in tqdm(os.listdir(root), unit="series"):
                if sid not in self.label_map: continue
                self.vols.append(preprocess_series(os.path.join(root, sid)))
                self.targets.append(self.label_map[sid])
                self.ids.append(sid)
            joblib.dump((self.vols, self.targets, self.ids), cache, compress=3)
            print(f"[INFO] cached tensors → {cache}")

    def __len__(self): return len(self.vols)
    def __getitem__(self, idx):
        return self.vols[idx], torch.tensor(self.targets[idx], dtype=torch.long)


def pad_collate(batch):
    max_d = max(v.shape[1] for v, _ in batch)
    vols, lbls = [], []
    for v, l in batch:
        pad = torch.zeros(3, max_d, *v.shape[2:], dtype=v.dtype)
        pad[:, :v.shape[1]] = v
        vols.append(pad); lbls.append(l)
    return torch.stack(vols), torch.stack(lbls)

# ╭───────────────────────────────────────────────────────────────────╮
# │  Model                                                           │
# ╰───────────────────────────────────────────────────────────────────╯
class Fisher3DCNN(nn.Module):
    def __init__(self, n_cls=NUM_CLASSES):
        super().__init__()
        self.backbone = r3d_18(weights="KINETICS400_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_cls)
    def forward(self, x): return self.backbone(x)

# ╭───────────────────────────────────────────────────────────────────╮
# │  Evaluation helpers                                              │
# ╰───────────────────────────────────────────────────────────────────╯
@torch.inference_mode()
def confusion_fig(model, loader, device, out="confusion_matrix.png"):
    y_true, y_pred = [], []
    for x, y in loader:
        y_true.extend(y.numpy())
        y_pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion")
    thr = cm.max()/2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>thr else "black")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"[INFO] confusion matrix → {out}")


def save_examples(ds: CTSeriesDataset, dest="examples", k=8):
    os.makedirs(dest, exist_ok=True)
    idxs = random.sample(range(len(ds)), k=min(k,len(ds)))
    for i in idxs:
        vol, lbl = ds[i]; series_id = ds.ids[i]
        mid = vol.shape[1]//2
        fig,axs = plt.subplots(1,3,figsize=(6,2))
        for c in range(3):
            axs[c].imshow(vol[c,mid], cmap="gray"); axs[c].axis("off"); axs[c].set_title(f"WL{c+1}")
        plt.suptitle(f"{series_id} | label {lbl.item()+1}")
        fn = os.path.join(dest, f"example_{series_id}.png")
        plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
        print(f"[INFO] example slice → {fn}")

# ╭───────────────────────────────────────────────────────────────────╮
# │  Training loop                                                   │
# ╰───────────────────────────────────────────────────────────────────╯
@torch.inference_mode()
def _avg_loss(model, loader, crit, device):
    s = 0.0
    for x,y in loader:
        s += crit(model(x.to(device)), y.to(device)).item()
    return s/len(loader)


def train(model, tr_loader, va_loader, epochs, lr, device):
    opt, crit = torch.optim.Adam(model.parameters(), lr), nn.CrossEntropyLoss()
    best, tr_losses, va_losses = 1e9, [], []

    for ep in range(1, epochs+1):
        model.train(); running=0.0
        bar = tqdm(tr_loader, desc=f"Ep {ep}/{epochs}", leave=False)
        for x,y in bar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x),y); loss.backward(); opt.step()
            running += loss.item(); bar.set_postfix(loss=running/(bar.n or 1))
        tr_loss = running/len(tr_loader); tr_losses.append(tr_loss)

        model.eval(); va_loss = _avg_loss(model, va_loader, crit, device); va_losses.append(va_loss)
        print(f"Ep {ep:4d}: train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), "best_3dcnn.pth")
            print("  ↳ new best saved 🏅")

    # save loss curves -------------------------------------------------------
    np.savetxt("train_losses.txt", tr_losses, fmt="%.6f")
    np.savetxt("val_losses.txt",   va_losses, fmt="%.6f")
    plt.figure(); plt.plot(tr_losses, label="train"); plt.plot(va_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("loss_curve.png", dpi=300); plt.close()
    print("[INFO] loss curve → loss_curve.png  &  txt logs saved")
    print(f"[DONE] best val loss = {best:.4f}")

# ╭───────────────────────────────────────────────────────────────────╮
# │  Main                                                             │
# ╰───────────────────────────────────────────────────────────────────╯

def parse_args():
    ap = argparse.ArgumentParser("Train 3-D CNN on skull-stripped CT data")
    ap.add_argument('--data_dir',   default='/home/ec2-user/Fisher/FlattenedDataset')
    ap.add_argument('--labels_csv', default='/home/ec2-user/Fisher/labels.csv')
    ap.add_argument('--batch',      type=int, default=2)
    ap.add_argument('--epochs',     type=int, default=500)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--regen_cache', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False

    ds = CTSeriesDataset(args.data_dir, args.labels_csv, regen=args.regen_cache)
    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=0.2, stratify=ds.targets, random_state=42)
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True,  collate_fn=pad_collate)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = Fisher3DCNN().to(device)
    print(f"[INFO] device={device} | train/val={len(tr_idx)}/{len(va_idx)} | epochs={args.epochs}")

    train(model, tr_loader, va_loader, args.epochs, args.lr, device)

    # post-training diagnostics --------------------------------------------
    model.load_state_dict(torch.load("best_3dcnn.pth", map_location=device)); model.eval()
    confusion_fig(model, va_loader, device)
    save_examples(ds)


if __name__ == '__main__':
    main()
