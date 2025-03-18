import re
import pandas as pd

file_path = "stoopid.txt" 

with open(file_path, "r") as file:
    log_text = file.read()

patient_ids = set(re.findall(r'Zach - Scored/(ID_[a-f0-9]+)/', log_text))

df_patients = pd.DataFrame(sorted(patient_ids), columns=['Unique Patient IDs'])

df_patients.to_csv("unique_patients_not_saved.csv", index=False)

print(df_patients)
