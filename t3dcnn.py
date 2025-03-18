import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


torch.backends.cudnn.enabled = False


# ---------------------
# Hyperparameters
# ---------------------
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 4  # Fisher scores: 0,1,2,3

# ---------------------
# Dataset Definition
# ---------------------
class InMemoryCTDataset(Dataset):
    """
    Expects a list of tensors [num_slices, 3, H, W] and a list of labels.
    """
    def __init__(self, images_list, labels_list):
        self.images_list = images_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        images = self.images_list[index]  
        label = self.labels_list[index]
        return images, torch.tensor(label, dtype=torch.long)

# ---------------------
#  Collate Function
# ---------------------
def pad_collate_3dcnn(batch):

    max_depth = max(sample[0].shape[0] for sample in batch)

    padded_images_list = []
    labels_list = []

    for (images, label) in batch:
        depth = images.shape[0] 

        images_t = images.permute(1, 0, 2, 3)  


        _, H, W = images_t.shape[1], images_t.shape[2], images_t.shape[3]
        padded = torch.zeros((3, max_depth, H, W), dtype=images.dtype)

        padded[:, :depth, :, :] = images_t
        padded_images_list.append(padded)
        labels_list.append(label)
    

    batch_images = torch.stack(padded_images_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)

    return batch_images, batch_labels

# ---------------------
# 3D CNN Model
# ---------------------
class Fisher3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Fisher3DCNN, self).__init__()
        self.backbone = models.r3d_18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ---------------------
# Training Function
# ---------------------
def train_3dcnn_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = "best_3dcnn_model.pth"

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device).float()  
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).float()
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    print("[INFO] Training complete. Best model path:", best_model_path)
    return train_losses, val_losses

# ---------------------
# Confusion Matrix
# ---------------------
def evaluate_and_plot_confusion(model, val_loader, class_names=None):
    if class_names is None:
        class_names = [str(i+1) for i in range(NUM_CLASSES)]  # '1','2','3','4'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig("3dcnn_confusion_matrix.png")
    plt.close()
    print("[INFO] Confusion matrix saved as 3dcnn_confusion_matrix.png")

# ---------------------
# Main Execution
# ---------------------
if __name__ == "__main__":

    dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"
    all_images, all_labels = joblib.load(dataset_cache_path)
    print(f"[INFO] Loaded dataset with {len(all_images)} series total.")

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    print(f"[INFO] Training set: {len(train_images)} series, Validation set: {len(val_images)} series")

    train_dataset = InMemoryCTDataset(train_images, train_labels)
    val_dataset   = InMemoryCTDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_3dcnn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_3dcnn
    )

    model = Fisher3DCNN(num_classes=NUM_CLASSES)
    

    train_losses, val_losses = train_3dcnn_model(
        model, train_loader, val_loader,
        epochs=EPOCHS,
        lr=LR
    )

    best_model_path = "best_3dcnn_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    evaluate_and_plot_confusion(model, val_loader, class_names=["1","2","3","4"])

    print("[INFO] Done!")
