import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk


series_path = "/home/ec2-user/Fisher/Fisher/FlattenedDataset/ID_008d06cb2d"
save_path = "comparison_skullstrip_fullseries.png"
WL, WW = 40, 80
HU_MIN, HU_MAX = 10, 50


def apply_window(hu_img, center, width):
    lower = center - (width / 2)
    upper = center + (width / 2)
    windowed = np.clip(hu_img, lower, upper)
    return ((255 * (windowed - lower) / width)).astype(np.uint8)

def get_pixel_based_mask(slice_hu):
    mask = (slice_hu > HU_MIN) & (slice_hu < HU_MAX)
    filled = binary_fill_holes(mask)
    closed = binary_closing(filled, footprint=np.ones((5, 5)))
    labeled = label(closed)
    if labeled.max() == 0:
        return np.zeros_like(closed)
    regions = regionprops(labeled)
    center = np.array(closed.shape) // 2
    best_region = min(regions, key=lambda r: np.linalg.norm(np.array(r.centroid) - center))
    return (labeled == best_region.label).astype(np.uint8)

def load_dicom_series(path):
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".dcm")]
    files.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    volume = np.stack([f.pixel_array for f in files])
    intercept = files[0].RescaleIntercept
    slope = files[0].RescaleSlope
    return volume * slope + intercept

def get_simpleitk_stripped_volume(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    brain_mask = sitk.BinaryThreshold(image, HU_MIN, HU_MAX, 1, 0)
    cc = sitk.ConnectedComponent(brain_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    largest_brain = sitk.BinaryThreshold(cc, largest_label, largest_label, 1, 0)
    return sitk.GetArrayFromImage(sitk.Mask(image, largest_brain))  # [z, y, x]


pixel_volume = load_dicom_series(series_path)
sitk_volume = get_simpleitk_stripped_volume(series_path)
num_slices = pixel_volume.shape[0]

fig, axs = plt.subplots(num_slices, 3, figsize=(12, num_slices * 2))

for i in range(num_slices):
    hu_slice = pixel_volume[i]
    windowed = apply_window(hu_slice, WL, WW)

    # Pixel-based
    pixel_mask = get_pixel_based_mask(hu_slice)
    pixel_stripped = windowed * pixel_mask


    sitk_hu = np.clip(sitk_volume[i], HU_MIN, HU_MAX)  # prevent overly bright background
    sitk_stripped = apply_window(sitk_hu, WL, WW)

    axs[i][0].imshow(windowed, cmap='gray')
    axs[i][0].set_title(f"Original Slice {i}")
    axs[i][0].axis('off')

    axs[i][1].imshow(pixel_stripped, cmap='gray')
    axs[i][1].set_title("Pixel-Based")
    axs[i][1].axis('off')

    axs[i][2].imshow(sitk_stripped, cmap='gray')
    axs[i][2].set_title("SimpleITK")
    axs[i][2].axis('off')

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

