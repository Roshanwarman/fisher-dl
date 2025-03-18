import os
import shutil
import pandas as pd
import pydicom
import traceback
import warnings

warnings.simplefilter("ignore", UserWarning)

# Define paths
base_dir = "/home/ec2-user/Fisher/Zach-Scored"  # Update this with the actual path
output_dir = "/home/ec2-user/Fisher/Data"  # Folder where organized series will be stored
csv_path = "/home/ec2-user/Fisher/ZH_labels.csv"  # Update this with the actual CSV path

def organize_dicom_files():
    # Load ground truth CSV
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)  # Handle missing values

    # Dictionary to map SeriesInstanceUID to fisher score
    series_fisher_scores = {}
    
    for patient_id in os.listdir(base_dir):
        patient_folder = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_folder):
            continue

        series_dict = {}
        series_counts = {}
        
        for dicom_file in os.listdir(patient_folder):
            dicom_path = os.path.join(patient_folder, dicom_file)
            
            try:
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                series_uid = str(ds.get("SeriesInstanceUID", ""))  # Ensure it's a string
                
                if series_uid not in series_dict:
                    series_dict[series_uid] = []
                    series_counts[series_uid] = 0
                series_dict[series_uid].append(dicom_path)
                series_counts[series_uid] += 1
            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")
                traceback.print_exc()

        
        for series_uid, file_list in series_dict.items():
            series_folder = os.path.join(output_dir, patient_id, series_uid)
            os.makedirs(series_folder, exist_ok=True)
            
            for file_path in file_list:
                shutil.copy(file_path, os.path.join(series_folder, os.path.basename(file_path)))  # Copy instead of move
            
            # Find matching fisher score for this patient and series by checking " image set"
            patient_data = df[df['Patient ID'] == patient_id]
            if not patient_data.empty:
                best_match = None
                min_diff = float("inf")
                
                for _, row in patient_data.iterrows():
                    expected_image_count = row['Notes'].split(" image set")[0].split("(")[-1] if "image set" in str(row['Notes']) else None
                    
                    if expected_image_count:
                        expected_image_count = int(expected_image_count)
                        diff = abs(series_counts[series_uid] - expected_image_count)
                        if diff < min_diff:
                            min_diff = diff
                            best_match = row['mFS']
                
                                
                if best_match is not None:
                    series_fisher_scores[series_uid] = best_match
                    print(f'{patient_id}  {series_uid}: {best_match} UNIU QUWUUEUE')
    
                # If no "image set" match was found, use patient ID's fisher score
                if best_match is None and len(patient_data) == 1:
                    best_match = patient_data.iloc[0]['mFS']
                    series_fisher_scores[series_uid] = best_match

                    print(f'{patient_id} {series_uid}: {best_match}')


    # Create new CSV mapping SeriesInstanceUID to fisher score
    output_csv = os.path.join(output_dir, "series_fisher_scores.csv")
    with open(output_csv, "w") as f:
        f.write("SeriesInstanceUID,FisherScore\n")
        for series_uid, fisher_score in series_fisher_scores.items():
            f.write(f"{series_uid},{fisher_score}\n")
    
    print(f"Organization complete. Data saved in {output_dir}")

if __name__ == "__main__":
    organize_dicom_files()
