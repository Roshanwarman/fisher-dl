import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np

dicom_dir = "/home/ec2-user/Fisher/Fisher/FlattenedDataset/ID_008d06cb2d"
output_nii = "skullstripped_brain.nii.gz"
output_png = "skullstripped_preview.png"
HU_MIN, HU_MAX = 10, 50
NUM_SLICES = 5  # Number of slices to plot

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_names)
image = reader.Execute()  #????

brain_mask = sitk.BinaryThreshold(image, lowerThreshold=HU_MIN, upperThreshold=HU_MAX, insideValue=1, outsideValue=0)

cc = sitk.ConnectedComponent(brain_mask)
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)
largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
largest_brain = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)

stripped_image = sitk.Mask(image, largest_brain)

sitk.WriteImage(stripped_image, output_nii)
print(f"Skull-stripped brain saved to: {output_nii}")

arr = sitk.GetArrayFromImage(stripped_image)  # Shape: [z, y, x]
num_slices = arr.shape[0]

fig, axs = plt.subplots(num_slices, 1, figsize=(5, num_slices * 2))

if num_slices == 1:
    axs = [axs]  

for i in range(num_slices):
    axs[i].imshow(arr[i], cmap='gray')
    axs[i].set_title(f"Slice {i}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()
print(f"Full series preview saved to: {output_png}")
