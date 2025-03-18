import re
import pandas as pd

# Define the file path
file_path = "stoopid.txt"  # Make sure this file exists in the same directory where you're running the script

# Read the content of the file
with open(file_path, "r") as file:
    log_text = file.read()

# Extract unique patient IDs (second element in the path)
patient_ids = set(re.findall(r'Zach - Scored/(ID_[a-f0-9]+)/', log_text))

# Convert to DataFrame for easy viewing
df_patients = pd.DataFrame(sorted(patient_ids), columns=['Unique Patient IDs'])

# Save to a CSV file (optional)
df_patients.to_csv("unique_patients_not_saved.csv", index=False)

# Display the results
print(df_patients)
