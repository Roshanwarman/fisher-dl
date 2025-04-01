import os
import joblib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Random3DTransforms:
    def __init__(self, p_flip=0.5, p_vflip=0.5, p_rot=0.5):
        self.p_flip = p_flip
        self.p_vflip = p_vflip
        self.p_rot = p_rot

    def __call__(self, volume):
        # volume: [T, C, H, W]
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
        images = self.images_list[index]  #
        label = self.labels_list[index]

        if self.transform is not None:
            images = self.transform(images)

        return images, torch.tensor(label, dtype=torch.long)

# ---------------------

# ---------------------
def visualize_3d_augmentations(dataset, transform, save_dir="aug_examples", num_samples=3):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        volume, label = dataset[idx] 
        
        volume_aug = transform(volume.clone())


        num_slices_to_show = 3
        if volume.shape[0] >= num_slices_to_show:
            slice_indices = np.linspace(0, volume.shape[0]-1, num_slices_to_show, dtype=int)
        else:
            slice_indices = list(range(volume.shape[0]))

        fig, axes = plt.subplots(2, len(slice_indices), figsize=(5 * len(slice_indices), 8))
        fig.suptitle(f"Sample index: {idx}, Label: {label}", fontsize=16)

        for j, s in enumerate(slice_indices):

            orig_slice = volume[s].permute(1, 2, 0).cpu().numpy()
            aug_slice  = volume_aug[s].permute(1, 2, 0).cpu().numpy()

            orig_slice = orig_slice.astype(np.float32)
            aug_slice  = aug_slice.astype(np.float32)

            orig_slice /= 255.0
            aug_slice  /= 255.0

            is_grayscale = (orig_slice.shape[2] == 1)

            axes[0, j].imshow(orig_slice, cmap='gray' if is_grayscale else None)
            axes[0, j].set_title(f"Original (slice {s})")
            axes[0, j].axis('off')

            axes[1, j].imshow(aug_slice, cmap='gray' if is_grayscale else None)
            axes[1, j].set_title(f"Augmented (slice {s})")
            axes[1, j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i}_label_{label}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved augmentation visualization: {save_path}")


if __name__ == "__main__":
    dataset_cache_path = "/home/ec2-user/Fisher/preloaded_dataset.pkl"
    all_images, all_labels = joblib.load(dataset_cache_path)

   
    dataset_no_transform = InMemoryCTDataset(all_images, all_labels, transform=None)

    random_3d_transform = Random3DTransforms(
        p_flip=0.5,
        p_vflip=0.5,
        p_rot=0.5
    )

    visualize_3d_augmentations(dataset_no_transform, random_3d_transform, 
                               save_dir="aug_examples", num_samples=5)

    print("[INFO] Visualization complete.")
