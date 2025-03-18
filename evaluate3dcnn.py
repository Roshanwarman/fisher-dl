import os
import joblib
import torch
import torch.nn as nn
import torchvision.models.video as models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) Dataset & Collate
# ---------------------------
class InMemoryCTDataset(torch.utils.data.Dataset):
    """
    Expects a list of tensors [depth, 3, H, W] and a list of labels.
    """
    def __init__(self, images_list, labels_list):
        self.images_list = images_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        images = self.images_list[index]  # shape [depth, 3, H, W]
        label = self.labels_list[index]   # int (Fisher score 0..3)
        return images, torch.tensor(label, dtype=torch.long)


def pad_collate_3dcnn(batch):
    """
    Each element in `batch` is (images, label) with images of shape [depth, 3, H, W].
    We pad these to the maximum depth in the batch:
      final shape = [B, 3, max_depth, H, W]
    """
    max_depth = max(sample[0].shape[0] for sample in batch)

    padded_images_list = []
    labels_list = []

    for (images, label) in batch:
        depth = images.shape[0]
        # (depth, 3, H, W) -> (3, depth, H, W)
        images_t = images.permute(1, 0, 2, 3)

        # Create zero-padded volume
        _, H, W = images_t.shape[1], images_t.shape[2], images_t.shape[3]
        padded = torch.zeros((3, max_depth, H, W), dtype=images.dtype)
        padded[:, :depth, :, :] = images_t

        padded_images_list.append(padded)
        labels_list.append(label)
    
    batch_images = torch.stack(padded_images_list, dim=0)  # (B, 3, max_depth, H, W)
    batch_labels = torch.stack(labels_list, dim=0)
    return batch_images, batch_labels

# ---------------------------
# 2) Model Definition
# ---------------------------
class Fisher3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Fisher3DCNN, self).__init__()
        self.backbone = models.r3d_18(pretrained=False)  
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, 3, depth, H, W)
        return self.backbone(x)

# ---------------------------
# 3) Utilities for Evaluation
# ---------------------------
def make_directory_if_needed(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.title("Validation set (N=67 scans)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")

def plot_prf_bars(report_dict, class_names, save_dir):
    """
    Plots bar charts for precision, recall, f1-score for each class.
    """
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        values = []
        for idx in range(len(class_names)):
            cls_key = str(idx)  # classification_report keys are strings
            if cls_key in report_dict:
                values.append(report_dict[cls_key][metric])
            else:
                values.append(0.0)
        
        plt.figure()
        plt.bar(class_names, values, color='blue')
        plt.title(f"{metric.capitalize()} per Class")
        plt.xlabel("Class")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        save_path = os.path.join(save_dir, f"{metric}_bar_chart.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved {metric} bar chart to {save_path}")

def evaluate_and_save_figs(model, dataloader, outdir, class_names=None):
    """
    1) Compute predictions -> accuracy, classification report, confusion matrix
    2) Save confusion matrix figure
    3) Save bar charts for precision, recall, f1
    4) Save classification report to text file
    """
    if class_names is None:
        class_names = [f"Class_{i+1}" for i in range(4)]

    make_directory_if_needed(outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")

    # Classification report
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, output_dict=False)
    print("\nClassification Report:\n", report_str)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(outdir, "confusion_matrix.png"))

    # Bar charts
    plot_prf_bars(report_dict, class_names, outdir)

    # Save text report
    report_txt_path = os.path.join(outdir, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n")
    print(f"[INFO] Saved classification report to {report_txt_path}")


# ---------------------------
# 4) Main: Split, Evaluate
# ---------------------------
if __name__ == "__main__":
    # Path to your single preloaded dataset
    dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"
    
    # 1) Load the entire dataset
    all_images, all_labels = joblib.load(dataset_cache_path)
    print(f"[INFO] Loaded dataset with {len(all_images)} total samples.")

    # 2) Split into train & valid sets (e.g. 80/20)
    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train set: {len(train_images)} samples, Valid set: {len(val_images)} samples")

    train_dataset = InMemoryCTDataset(train_images, train_labels)
    val_dataset   = InMemoryCTDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=False,
        collate_fn=pad_collate_3dcnn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=pad_collate_3dcnn
    )

    # 3) Load model & weights
    model = Fisher3DCNN(num_classes=4)
    best_weights_path = "best_3dcnn_model.pth"
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
    print(f"[INFO] Loaded model weights from {best_weights_path}")

    # 4) Evaluate on the TRAIN set
    print("\n===== EVALUATING ON TRAIN SET =====")
    evaluate_and_save_figs(
        model, train_loader,
        outdir="train",           # Folder to save figures/reports
        class_names=["1","2","3","4"]
    )

    # 5) Evaluate on the VALIDATION set
    print("\n===== EVALUATING ON VALIDATION SET =====")
    evaluate_and_save_figs(
        model, val_loader,
        outdir="valid",           # Folder to save figures/reports
        class_names=["1","2","3","4"]
    )

    print("\n[INFO] Done! Check 'train/' and 'valid/' folders for saved figures and classification reports.")
