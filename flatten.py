import os
import shutil

original_dataset_path = "/home/ec2-user/Fisher/Data"
flattened_dataset_path = "/home/ec2-user/Fisher/FlattenedDataset"

os.makedirs(flattened_dataset_path, exist_ok=True)

for patient_id in os.listdir(original_dataset_path):
    patient_path = os.path.join(original_dataset_path, patient_id)

    if os.path.isdir(patient_path): 
        for series_id in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series_id)
            if os.path.isdir(series_path):  
                target_path = os.path.join(flattened_dataset_path, series_id)
                shutil.move(series_path, target_path)

print("Flattening complete.")
