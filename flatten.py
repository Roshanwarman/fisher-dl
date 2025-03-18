import os
import shutil

# Define paths
original_dataset_path = "/home/ec2-user/Fisher/Data"
flattened_dataset_path = "/home/ec2-user/Fisher/FlattenedDataset"

# Ensure the new directory exists
os.makedirs(flattened_dataset_path, exist_ok=True)

# Traverse the original dataset and move seriesIDs to the new folder
for patient_id in os.listdir(original_dataset_path):
    patient_path = os.path.join(original_dataset_path, patient_id)

    if os.path.isdir(patient_path):  # Ensure it's a directory
        for series_id in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series_id)
            if os.path.isdir(series_path):  # Ensure it's a directory
                target_path = os.path.join(flattened_dataset_path, series_id)
                shutil.move(series_path, target_path)

print("Flattening complete.")
