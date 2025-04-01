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
from torch.utils.data.sampler import WeightedRandomSampler


torch.backends.cudnn.enabled = False


# ---------------------
# Hyperparameters
# ---------------------
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 4 

import random
import torch

class Random3DTransforms:

    def __init__(self, p_flip=0.5, p_vflip=0.5, p_rot=0.5):
        self.p_flip = p_flip    
        self.p_vflip = p_vflip  
        self.p_rot = p_rot       

    def __call__(self, volume):
        
        if random.random() < self.p_flip:
            volume = torch.flip(volume, dims=[3])  
        
        if random.random() < self.p_vflip:
            volume = torch.flip(volume, dims=[2])  
        
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            volume = torch.rot90(volume, k, dims=[2, 3])
        
        return volume

class InMemoryCTDataset(Dataset):
    def __init__(self, images_list, labels_list, transform=None):
        self.images_list = images_list
        self.labels_list = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        images = self.images_list[index]  # [T, 3, H, W]
        label = self.labels_list[index]

        if self.transform is not None:
            images = self.transform(images)

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
    best_model_path = "best3dcnn_aug.pth"

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


if __name__ == "__main__":
    dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"
    all_images, all_labels = joblib.load(dataset_cache_path)
    print(f"[INFO] Loaded dataset with {len(all_images)} series total.")

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.4, random_state=42
    )

    train_transform = Random3DTransforms(p_flip=0.5, p_vflip=0.3, p_rot=0.3)
    val_transform   = None  

    train_dataset = InMemoryCTDataset(train_images, train_labels, transform=train_transform)
    val_dataset   = InMemoryCTDataset(val_images, val_labels, transform=val_transform)


    train_label_arr = np.array(train_labels)
    class_sample_counts = np.bincount(train_label_arr) 

    weights_per_class = 1.0 / (class_sample_counts + 1e-6)
    sample_weights = weights_per_class[train_label_arr]
    sample_weights = torch.from_numpy(sample_weights).float()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=pad_collate_3dcnn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_3dcnn
    )

    model = Fisher3DCNN(num_classes=NUM_CLASSES)
    best_model_path = "best3dcnn_aug.pth"

    try:
    
        train_losses, val_losses = train_3dcnn_model(
            model, train_loader, val_loader,
            epochs=EPOCHS,
            lr=LR
        )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Proceeding to evaluation of best model (if available)...")
    finally:
        try:
            if os.path.exists(best_model_path):
                best_model_path = "best3dcnn_aug.pth"
                model.load_state_dict(torch.load(best_model_path))
                evaluate_and_plot_confusion(model, val_loader, class_names=["1","2","3","4"])

            else:
                print("[WARNING] No best model found to evaluate.")
        except KeyboardInterrupt:
            print("\n[INFO] Evaluation interrupted. Exiting cleanly.")

    print("[INFO] Done!")


    # print(f'{train_loader.shape} yoooo test')
    # print("[testo what's up my name is roshanhello?")

    # train_losses, val_losses = train_3dcnn_model(
    #     model, train_loader, val_loader,
    #     epochs=EPOCHS,
    #     lr=LR
    # )