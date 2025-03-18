import os
import fnmatch
import pydicom
import cv2
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")  # So it can run headless on servers
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

# -------------------------
# 1) Brain Window Preprocessing
# -------------------------

def apply_brain_window(img, wl=40, ww=80):
    """
    Converts raw HU slice into 1-channel image using Brain Window.
    wl=40 (window level), ww=80 (window width)
    Returns a uint8 image in [0..255].
    """
    lower = wl - (ww / 2.)
    upper = wl + (ww / 2.)
    img_clipped = np.clip(img, lower, upper)
    # Scale to [0..255]
    img_scaled = 255 * (img_clipped - lower) / (ww)
    img_scaled = img_scaled.astype(np.uint8)
    return img_scaled

def load_dicom_and_window(dcm_path):
    """
    Reads one DICOM file and applies Brain window to produce a single-channel uint8 image.
    """
    dcm = pydicom.dcmread(dcm_path)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)

    hu = (dcm.pixel_array.astype(np.float32) * slope) + intercept
    brain_img = apply_brain_window(hu, wl=40, ww=80)  # single channel in grayscale
    return brain_img

def process_dicom_series(series_folder, target_size=224):
    """
    Reads all .dcm files in a series folder, sorts them by descending z,
    and returns a tensor of shape [num_slices, 1, H, W].
    """
    dcm_files = fnmatch.filter(os.listdir(series_folder), "*.dcm")
    if not dcm_files:
        return None  # skip empty or invalid

    # Read and sort by z-position descending
    dicoms = []
    for f in dcm_files:
        dcm_path = os.path.join(series_folder, f)
        ds = pydicom.dcmread(dcm_path)
        dicoms.append(ds)
    dicoms.sort(key=lambda x: -x.ImagePositionPatient[2])

    # Apply Brain window to each slice
    slices = []
    for ds in dicoms:
        slope = getattr(ds, "RescaleSlope", 1)
        intercept = getattr(ds, "RescaleIntercept", 0)
        raw_hu = ds.pixel_array.astype(np.float32) * slope + intercept
        bw_img = apply_brain_window(raw_hu, wl=40, ww=80)  # shape (H, W) in uint8

        # Resize to target_size
        bw_img = cv2.resize(bw_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Convert to tensor shape [1, H, W] (1-channel)
        bw_img_t = torch.as_tensor(bw_img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        slices.append(bw_img_t)

    if not slices:
        return None

    # Stack along slice dimension -> shape [num_slices, 1, H, W]
    all_slices = torch.stack(slices, dim=0)
    return all_slices

# -------------------------
# 2) Dataset & Collate
# -------------------------

class CTBrainWindowDataset(Dataset):
    """
    Loads each series folder, applies Brain window to each slice => single-channel slices.
    Expects a CSV with `series_id` and `score` columns for Fisher score.
    """
    def __init__(self, root_dir, label_csv):
        super().__init__()
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(label_csv)
        # Make a dict: series_id -> label
        self.labels_map = dict(zip(self.labels_df["series_id"], self.labels_df["score"]))

        # We'll gather all series folders in root_dir
        self.series_folders = []
        for series_name in os.listdir(root_dir):
            path = os.path.join(root_dir, series_name)
            if os.path.isdir(path):
                # Check if we have a label
                if series_name in self.labels_map:
                    self.series_folders.append(series_name)
                else:
                    print(f"[WARNING] No label found for {series_name}, skipping.")

    def __len__(self):
        return len(self.series_folders)

    def __getitem__(self, idx):
        series_name = self.series_folders[idx]
        series_path = os.path.join(self.root_dir, series_name)

        # Load slices using only Brain window
        all_slices = process_dicom_series(series_path)
        if all_slices is None:
            # In practice, you might skip or handle an error
            raise RuntimeError(f"No valid slices found in {series_path}.")

        # label: convert 1..4 => 0..3
        label = self.labels_map[series_name] - 1

        return all_slices, torch.tensor(label, dtype=torch.long)

def pad_collate_3dcnn(batch):
    """
    We have a list of (images, label) with images of shape [depth, 1, H, W].
    We pad to the max depth in the batch -> [B, 1, max_depth, H, W].
    """
    max_depth = max(sample[0].shape[0] for sample in batch)
    padded_images_list = []
    labels_list = []

    for (images, label) in batch:
        depth = images.shape[0]  # number of slices
        # shape: (depth, 1, H, W) -> (1, depth, H, W)
        images_t = images.permute(1, 0, 2, 3)  # now (1, depth, H, W)

        # Pad to max_depth
        _, H, W = images_t.shape[1], images_t.shape[2], images_t.shape[3]
        padded = torch.zeros((1, max_depth, H, W), dtype=images.dtype)
        padded[:, :depth, :, :] = images_t

        padded_images_list.append(padded)
        labels_list.append(label)

    batch_images = torch.stack(padded_images_list, dim=0)  # (B, 1, max_depth, H, W)
    batch_labels = torch.stack(labels_list, dim=0)
    return batch_images, batch_labels

# -------------------------
# 3) Single-channel 3D CNN
# -------------------------

class SingleChannel3DCNN(nn.Module):
    """
    We adapt r3d_18 to accept 1-channel input instead of 3.
    """
    def __init__(self, num_classes=4):
        super(SingleChannel3DCNN, self).__init__()
        self.backbone = r3d_18(pretrained=True)  # If you want Kinetics pretrained
        # Modify the first convolution layer to accept 1 channel
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1, out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        # Replace final FC for Fisher classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: (B, 1, depth, H, W)
        return self.backbone(x)

# -------------------------
# 4) Training & Validation
# -------------------------

def train_3dcnn_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = "best_brain3dcnn_model.pth"
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device).float()  # (B, 1, depth, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).float()
                labels = labels.to(device)

                outputs = model(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    print(f"[INFO] Training complete. Best model saved to {best_model_path}")
    return train_losses, val_losses, best_model_path

def evaluate_model(model, loader, num_classes=4):
    """
    Returns (accuracy, confusion_matrix, classification_report_string)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, labels=range(num_classes), digits=4)
    accuracy = (cm.diagonal().sum() / cm.sum()) if cm.sum() > 0 else 0

    return accuracy, cm, report

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    # Text
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")

# -------------------------
# 5) Main
# -------------------------

if __name__ == "__main__":
    # Example usage: if you have a "FlattenedDataset" folder
    # with each series as a subfolder, and a label CSV
    dataset_path = "/home/ec2-user/Fisher/FlattenedDataset"  # adjust
    labels_csv   = "/home/ec2-user/Fisher/labels.csv"        # adjust
    BATCH_SIZE   = 2
    EPOCHS       = 5

    # 1) Create Dataset
    full_dataset = CTBrainWindowDataset(dataset_path, labels_csv)

    # 2) Split into Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Train set: {len(train_dataset)}    Val set: {len(val_dataset)}")

    # 3) Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=pad_collate_3dcnn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=pad_collate_3dcnn)

    # 4) Model
    model = SingleChannel3DCNN(num_classes=4)

    # 5) Train
    train_losses, val_losses, best_model_path = train_3dcnn_model(
        model, train_loader, val_loader, epochs=EPOCHS, lr=1e-4
    )

    # 6) Evaluate best model
    model.load_state_dict(torch.load(best_model_path))
    acc, cm, report = evaluate_model(model, val_loader, num_classes=4)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    class_names = ["Fisher1", "Fisher2", "Fisher3", "Fisher4"]
    plot_confusion_matrix(cm, class_names, save_path="brain_window_confusion_matrix.png")

    # 7) (Optional) Plot losses
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (Brain Window 3D CNN)")
    plt.legend()
    plt.grid()
    plt.savefig("brain_window_losses.png", dpi=300)
    print("[INFO] Saved train/val loss curve to brain_window_losses.png")
    plt.close()
