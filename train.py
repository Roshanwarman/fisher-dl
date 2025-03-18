import os
import fnmatch
import gc
import pandas as pd
import numpy as np
import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import pretrainedmodels
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

# -------------------------------
# Define Model with New Head
# -------------------------------

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        
        backbone = pretrainedmodels.__dict__["resnet50"](num_classes=1000, pretrained="imagenet")
        
        self.layer0 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu', nn.ReLU(inplace=True)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ])
        )
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def resnet_forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class FisherScoreHead(nn.Module):
    def __init__(self):
        super(FisherScoreHead, self).__init__()
        self.recurrent = nn.LSTM(
            input_size=2048, hidden_size=512, dropout=0.3, 
            num_layers=2, bidirectional=True, batch_first=True
        )
        
        self.fc = nn.Linear(1024, 4)  # New: 4 classes
    
    def forward(self, x, seq_len):
        # print('faslkjdflkajsdflas')
        # print(x.shape)

        x = F.adaptive_avg_pool2d(x, 1)  # Pooling to (B*Seq, 2048, 1, 1)

        x = x.view(x.shape[0], x.shape[1], -1)  # Now (B, Seq, 2048)

        batch_size = max(1, x.shape[0] // seq_len)  # Prevent batch_size from being 0
        feature_dim = x.shape[1]  # Should be 2048
        # print(x.shape)
        # Pass through LSTM
        x, _ = self.recurrent(x)

        x = self.fc(x)

        x = x[:, -1, :]  # Taking only the last output in sequence

        return x

class FisherModel(Resnet):
    def __init__(self):
        super(FisherModel, self).__init__()
        self.decoder = FisherScoreHead()

    def forward(self, x, seq_len):
        batch_size, seq_len, channels, height, width = x.shape  # Unpack dimensions
        
        x = x.view(batch_size * seq_len, channels, height, width)
        # print(x.shape)
        # Pass through ResNet backbone
        x = self.resnet_forward(x)

        x = x.view(batch_size, seq_len, x.shape[1], x.shape[2], x.shape[3])

        x = self.decoder(x, seq_len)

        return x


# -------------------------------
# DICOM Preprocessing
# -------------------------------

def preprocess_dicom(dcm, window_index):
    wl = [40, 75, 600]
    ww = [80, 215, 2800]
    window_min = wl[window_index] - (ww[window_index] // 2)
    window_max = wl[window_index] + (ww[window_index] // 2)

    img = (dcm.pixel_array * dcm.RescaleSlope) + dcm.RescaleIntercept
    img = np.clip(img, window_min, window_max)
    img = 255 * ((img - window_min) / ww[window_index])
    img = img.astype(np.uint8)
    return img

def load_img(img):
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.array(img).transpose(-1, 0, 1)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    
    return torch.Tensor(x)

def process_dicom_series(dicom_files, target_size=224):
    dicom_data = [pydicom.dcmread(f) for f in dicom_files]
    dicom_data = sorted(dicom_data, key=lambda x: -x.ImagePositionPatient[2])

    total_scan = []
    for dcm in dicom_data:
        imgs = [preprocess_dicom(dcm, i) for i in range(3)]
        rgb = np.dstack(imgs)  

        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        total_scan.append(rgb)

    images = [load_img(img) for img in total_scan]  
    return torch.stack(images)

# -------------------------------
# Dataset & DataLoader
# -------------------------------

class FisherDataset(Dataset):
    def __init__(self, data_dir, label_csv):
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_csv)
        
        self.series = []
        for patient in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient)
            if os.path.isdir(patient_path):
                for series in os.listdir(patient_path):
                    series_path = os.path.join(patient_path, series)
                    if os.path.isdir(series_path):
                        self.series.append(series_path)
        
        self.labels.set_index("series_id", inplace=True)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        series_path = self.series[idx]
        dicom_files = fnmatch.filter(os.listdir(series_path), "*.dcm")
        dicom_files = [os.path.join(series_path, f) for f in dicom_files]
        
        images = process_dicom_series(dicom_files)
        series_name = os.path.basename(series_path)
        label = self.labels.loc[series_name, "score"]
        label = label - 1  # now it 0,1,2,3 asdfasdfasdf
        return images, torch.tensor(label, dtype=torch.long)


class InMemoryCTDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, labels_list):
        self.images_list = images_list  
        self.labels_list = labels_list  

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        images = self.images_list[index]  
        label = self.labels_list[index]  
        
        return images, torch.tensor(label, dtype=torch.long)

# *-------------------
# collate function
# -------------------

def pad_collate(batch):

    max_seq_len = max(sample[0].shape[0] for sample in batch)
    # for i in batch:
        # print(i[0].shape[0])
    # print(f'largest sequence {max_seq_len}')
    padded_images_list = []
    labels_list = []
    lengths_list = []
    
    for (images, label) in batch:
        seq_len = images.shape[0]
        lengths_list.append(seq_len)

        padded = torch.zeros(
            (max_seq_len, 3, images.shape[2], images.shape[3]),
            dtype=images.dtype
        )
        
        padded[:seq_len] = images
        
        padded_images_list.append(padded)
        labels_list.append(label)

    batch_images = torch.stack(padded_images_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)
    batch_lengths = torch.tensor(lengths_list, dtype=torch.long)
    
    return batch_images, batch_labels, batch_lengths


# ---------------------------
# validation
# ----------------------------


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_and_plot_confusion(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, lengths in val_loader:
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model(images, lengths)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    class_names = ["1","2","3","4"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close(fig)
    print("Confusion matrix saved to confusion_matrix.png")




labels_csv_path = "/home/ec2-user/Fisher/labels.csv"


labels_df = pd.read_csv(labels_csv_path)
series_to_label = dict(zip(labels_df["series_id"], labels_df["score"]))

def get_label_for_series(series_folder):

    label = series_to_label.get(series_folder)  
    if label is None or pd.isna(label):  
        print(f"Warning: No label found for {series_folder}. Skipping entry.")
        return None  

    return int(label) - 1  

def load_full_dataset_in_memory(root_dir, target_size=224):
    all_images = []
    all_labels = []

    for series_folder in os.listdir(root_dir):
        series_path = os.path.join(root_dir, series_folder)
        if not os.path.isdir(series_path):
            continue

        label = get_label_for_series(series_folder)
        if label is None:  
            print(f"Skipping {series_folder} due to missing label.")
            continue  

        dcm_files = sorted(
            [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")],
            key=lambda x: -pydicom.dcmread(os.path.join(series_path, x)).ImagePositionPatient[2]
        )

        dicom_data = [pydicom.dcmread(os.path.join(series_path, f)) for f in dcm_files]

        series_slices = []
        for dcm in dicom_data:
            slices_3ch = [preprocess_dicom(dcm, i) for i in range(3)]
            rgb = np.dstack(slices_3ch)
            rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
            rgb_tensor = torch.as_tensor(rgb, dtype=torch.float16).permute(2, 0, 1)  # Convert to [3, H, W]
            series_slices.append(rgb_tensor)

        if series_slices:
            series_tensor = torch.stack(series_slices, dim=0)  # [num_slices, 3, H, W]
            all_images.append(series_tensor)
            all_labels.append(label)
        else:
            print(f"Skipping {series_folder} as it contains no valid DICOM slices.")

    return all_images, all_labels

# -------------------------------
# Training Setup
# -------------------------------

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = "best_model.pth"
    
    epoch_reached = 0  
    
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for images, labels, lengths in train_loader:
                images = images.to(device).to(torch.float32)
                labels = labels.to(device)
                lengths = lengths.to(device)

                optimizer.zero_grad()
                outputs = model(images, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels, lengths in val_loader:
                    images = images.to(device).to(torch.float32)
                    labels = labels.to(device)
                    lengths = lengths.to(device)

                    outputs = model(images, lengths)
                    val_loss += criterion(outputs, labels).item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            epoch_reached = epoch + 1
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Caught KeyboardInterrupt at epoch {epoch_reached}. Saving partial logs and exiting...")
    
    finally:

        with open("train_loss.txt", "w") as f:
            for tl in train_losses:
                f.write(f"{tl}\n")
        
        with open("val_loss.txt", "w") as f:
            for vl in val_losses:
                f.write(f"{vl}\n")
        

        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(range(1, epoch_reached+1), train_losses[:epoch_reached], 
                 label="Train Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig("train_loss.png")
        plt.close()

        plt.figure()
        plt.plot(range(1, epoch_reached+1), val_losses[:epoch_reached],
                 label="Validation Loss", marker="o", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig("val_loss.png")
        plt.close()
        

        # -----------------------------
        # -----------------------------
        print("[INFO] Loading best model and computing confusion matrix on validation set...")
        model.load_state_dict(torch.load(best_model_path))
        evaluate_and_plot_confusion(model, val_loader)
        
        print("[INFO] Training/validation logs saved. Confusion matrix saved. Done.")
    
    return model




# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    dataset_path = "/home/ec2-user/Fisher/FlattenedDataset"
    labels_csv_path = "/home/ec2-user/Fisher/labels.csv"


    # labels_df = pd.read_csv(labels_csv_path)
    # series_to_label = dict(zip(labels_df["series_id"], labels_df["score"]))
    # all_images, all_labels = load_full_dataset_in_memory(dataset_path)
    # print("Total series loaded:", len(all_images))


    import joblib

    # dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"
    # joblib.dump((all_images, all_labels), dataset_cache_path, compress=3)
    # print(f"Preloaded dataset saved/ at: {dataset_cache_path}")



    dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"

    all_images, all_labels = joblib.load(dataset_cache_path)

    print(f"Dataset loaded! Total series: {len(all_images)}")

    from sklearn.model_selection import train_test_split

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )


    print("Unique labels in training set:", set(train_labels))
    print("Unique labels in validation set:", set(val_labels))

    # exit()
    # Create Dataset objects
    train_dataset = InMemoryCTDataset(train_images, train_labels)
    val_dataset = InMemoryCTDataset(val_images, val_labels)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate
    )

    # dataset = FisherDataset(dataset_path, labels_csv)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=pad_collate)

    model = FisherModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_ckpt = torch.load("pretrain.pth", map_location=device)
    old_state_dict = old_ckpt["state_dict"]  # or just old_ckpt if stored directly

    new_state_dict = model.state_dict()

    filtered_dict = {}
    for k, v in old_state_dict.items():
        if (k in new_state_dict) and (new_state_dict[k].shape == v.shape):
            filtered_dict[k] = v

    new_state_dict.update(filtered_dict)

    model.load_state_dict(new_state_dict)


    matched_keys = list(filtered_dict.keys())
    print(f"[INFO] Loaded {len(matched_keys)} layers from old checkpoint.")
    missing_in_old = set(new_state_dict.keys()) - set(old_state_dict.keys())
    print(f"[INFO] Missing in old checkpoint: {missing_in_old}")
    not_loaded = set(old_state_dict.keys()) - set(filtered_dict.keys())
    print(f"[INFO] Not loaded (shape mismatch or new layer): {not_loaded}")

    model = train_model(model, train_loader, val_loader, epochs=200, lr=1e-4)

    # model.load_state_dict(torch.load("best_model.pth"))
    # evaluate_and_plot_confusion(model, val_loader)