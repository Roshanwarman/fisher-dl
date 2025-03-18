import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_window(img, window_center, window_width):

    lower = window_center - (window_width / 2)
    upper = window_center + (window_width / 2)
    
    # Clip
    windowed_img = np.clip(img, lower, upper)
    
    # Normalize to 0..255
    windowed_img = 255 * (windowed_img - lower) / (window_width)
    windowed_img = windowed_img.astype(np.uint8)
    return windowed_img

# Common window settings for CT head
BRAIN_WINDOW  = (40, 80)    # (WL=40, WW=80)
SUBDURAL_WINDOW = (75, 215) # (WL=75, WW=215)
BONE_WINDOW   = (600, 2800) # (WL=600, WW=2800)

if __name__ == "__main__":
    # 1) Load a DICOM file (adjust path)
    dcm_path = "/home/ec2-user/Fisher/FlattenedDataset/ID_0b50fe17d1/ID_d7a727d6a.dcm"
    dcm = pydicom.dcmread(dcm_path)

    # 2) Convert raw pixel data to Hounsfield units (HU)
    #    HU = pixel_array * RescaleSlope + RescaleIntercept
    raw_img = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    hu_img = (raw_img * slope) + intercept

    # 3) Generate three windowed versions
    brain_img    = apply_window(hu_img, BRAIN_WINDOW[0], BRAIN_WINDOW[1])      # WL=40,  WW=80
    subdural_img = apply_window(hu_img, SUBDURAL_WINDOW[0], SUBDURAL_WINDOW[1])# WL=75,  WW=215
    bone_img     = apply_window(hu_img, BONE_WINDOW[0], BONE_WINDOW[1])        # WL=600, WW=2800

    # 4) Plot side by side
    plt.figure(figsize=(12, 4))  # width=12 inches, height=4 inches

    plt.subplot(1, 3, 1)
    plt.imshow(brain_img, cmap="gray")
    plt.title("Brain Window\n(WL=40, WW=80)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(subdural_img, cmap="gray")
    plt.title("Subdural Window\n(WL=75, WW=215)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(bone_img, cmap="gray")
    plt.title("Bone Window\n(WL=600, WW=2800)")
    plt.axis("off")

    plt.tight_layout()
    
    # 5) Save high-resolution figure
    out_path = "windows_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved side-by-side window figure at high resolution: {out_path}")
    plt.show()
