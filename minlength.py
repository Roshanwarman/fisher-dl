import os

data_path = '/home/ec2-user/Fisher/Data'

max_count = 0
max_series_path = None

for patient in os.listdir(data_path):
    patient_path = os.path.join(data_path, patient)
    if os.path.isdir(patient_path):
        for series in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series)
            if os.path.isdir(series_path):
                dcm_files = [f for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
                dcm_count = len(dcm_files)
                
                if dcm_count > max_count:
                    max_count = dcm_count
                    max_series_path = series_path

print(f"The series folder with the most .dcm files is:\n  {max_series_path}")
print(f"Number of .dcm files: {max_count}")
