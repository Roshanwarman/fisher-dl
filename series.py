import os

def analyze_dicom_folders(root_dir):
    """
    Analyzes a directory structure to find subfolders with the most and fewest .dcm files,
    and prints the length of each subfolder.

    Args:
        root_dir (str): The root directory containing patient folders.
    """

    subfolder_counts = {}

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if os.path.isdir(patient_path):
            for subfolder in os.listdir(patient_path):
                subfolder_path = os.path.join(patient_path, subfolder)
                if os.path.isdir(subfolder_path):
                    dcm_files = [f for f in os.listdir(subfolder_path) if f.endswith(".dcm")]
                    subfolder_counts[subfolder_path] = len(dcm_files)

    if not subfolder_counts:
        print("No subfolders with .dcm files found.")
        return

    most_files_subfolder = max(subfolder_counts, key=subfolder_counts.get)
    least_files_subfolder = min(subfolder_counts, key=subfolder_counts.get)

    print("Subfolders and their .dcm file counts:")
    for subfolder, count in subfolder_counts.items():
        print(f"{subfolder}: {count}")

    print("\nSubfolder with the most .dcm files:")
    print(f"{most_files_subfolder}: {subfolder_counts[most_files_subfolder]}")

    print("\nSubfolder with the fewest .dcm files:")
    print(f"{least_files_subfolder}: {subfolder_counts[least_files_subfolder]}")

if __name__ == "__main__":
    root_directory = "/home/ec2-user/Fisher/Data"  # Replace with your root directory path
    analyze_dicom_folders(root_directory)