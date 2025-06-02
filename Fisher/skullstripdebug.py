import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes

# --- Config ---
WL, WW = 40, 80
HU_MIN, HU_MAX = 10, 50
series_path = "/home/ec2-user/Fisher/Fisher/FlattenedDataset/ID_008d06cb2d"
save_path = "debug_skullstrip_slices18_22.png"
slice_indices = list(range(1, 30))

def apply_window(hu_img, center, width):
    lower = center - (width / 2)
    upper = center + (width / 2)
    windowed = np.clip(hu_img, lower, upper)
    return ((255 * (windowed - lower) / width)).astype(np.uint8)

def get_mask_debug(slice_hu):
    debug = {}

    # Step 1: Threshold HU
    mask = (slice_hu > HU_MIN) & (slice_hu < HU_MAX)
    debug['thresholded_mask'] = mask.astype(np.uint8)

    # Step 2: Morphology
    filled = binary_fill_holes(mask)
    closed = binary_closing(filled, footprint=np.ones((5, 5)))
    debug['post_morph'] = closed.astype(np.uint8)

    # Step 3: Label regions
    labeled = label(closed)
    debug['labeled'] = labeled

    if labeled.max() == 0:
        debug['final_mask'] = np.zeros_like(closed)
        debug['selected_centroid'] = None
        return debug

    regions = regionprops(labeled)
    center = np.array(closed.shape) // 2

    # Choose region with centroid closest to image center
    best_region = min(regions, key=lambda r: np.linalg.norm(np.array(r.centroid) - center))
    mask_final = labeled == best_region.label

    debug['final_mask'] = mask_final.astype(np.uint8)
    debug['selected_centroid'] = best_region.centroid

    return debug

def load_series(path):
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".dcm")]
    files.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    volume = np.stack([f.pixel_array for f in files])
    intercept = files[0].RescaleIntercept
    slope = files[0].RescaleSlope
    return volume * slope + intercept

def plot_debug(volume, slice_indices, save_path):
    fig, axs = plt.subplots(len(slice_indices), 6, figsize=(20, len(slice_indices) * 3))

    for i, idx in enumerate(slice_indices):
        hu = volume[idx]
        windowed = apply_window(hu, WL, WW)
        debug = get_mask_debug(hu)
        final_mask = debug['final_mask']
        labeled = debug['labeled']
        centroid = debug['selected_centroid']

        # 1. Original HU
        axs[i][0].imshow(hu, cmap='gray')
        axs[i][0].set_title(f"HU Slice {idx}")
        axs[i][0].axis('off')

        # 2. Thresholded Mask
        axs[i][1].imshow(debug['thresholded_mask'], cmap='gray')
        axs[i][1].set_title("Thresholded HU")
        axs[i][1].axis('off')

        # 3. Post Morphology
        axs[i][2].imshow(debug['post_morph'], cmap='gray')
        axs[i][2].set_title("Morphology Applied")
        axs[i][2].axis('off')

        # 4. Labeled + Centroids
        axs[i][3].imshow(labeled, cmap='nipy_spectral')
        axs[i][3].set_title("Labeled Regions")
        if centroid:
            axs[i][3].plot(centroid[1], centroid[0], 'ro')
        axs[i][3].axis('off')

        # 5. Final Brain Mask
        axs[i][4].imshow(final_mask, cmap='gray')
        axs[i][4].set_title("Selected Region")
        axs[i][4].axis('off')

        # 6. Stripped Image
        stripped = windowed * final_mask
        axs[i][5].imshow(stripped, cmap='gray')
        axs[i][5].set_title("Skull-Stripped Output")
        axs[i][5].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🧠 Debug figure saved: {save_path}")

# --- Run ---
if __name__ == "__main__":
    vol = load_series(series_path)
    plot_debug(vol, slice_indices, save_path)
