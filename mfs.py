import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
df = pd.read_excel("SAH_mFS_Rai_MASTER_cpy.xlsx")

# Set plot style
sns.set(style="whitegrid")

# Plot the distribution of mFS scores
plt.figure(figsize=(10, 6), dpi=300)
sns.countplot(x='mFS', data=df, palette='viridis')
plt.title('Distribution of mFS Scores')
plt.xlabel('mFS Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('mFS_distribution.png')
plt.close()

# Plot counts for EVD, Craniotomy, and Clean
procedure_counts = df[['EVD', 'Craniotomy', 'Clean']].apply(lambda x: x.notna().sum())

plt.figure(figsize=(10, 6), dpi=300)
procedure_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Procedures')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('procedure_counts.png')
plt.close()

# Count scans per patient
scans_per_patient = df.groupby('Patient ID').size()

plt.figure(figsize=(12, 6), dpi=300)
sns.histplot(scans_per_patient, bins=range(1, scans_per_patient.max() + 2), kde=False, color='coral')
plt.title('Distribution of Scans per Patient')
plt.xlabel('Number of Scans')
plt.ylabel('Number of Patients')
plt.xticks(range(1, scans_per_patient.max() + 1))
plt.tight_layout()
plt.savefig('scans_per_patient.png')
plt.close()

print("Plots have been saved as high-resolution images.")
