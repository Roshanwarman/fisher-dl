import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
file_path = '/home/ec2-user/Fisher/labels.csv'
df = pd.read_csv(file_path)

# Count the occurrences of each score
score_counts = df['score'].value_counts().sort_index()

# Print the counts for scores 1 through 4
for score in range(1, 5):
    count = score_counts.get(score, 0)
    print(f"Number of score {score}s: {count}")

# Plot the histogram
plt.figure(figsize=(6, 4))
bars = plt.bar(score_counts.index.astype(str), score_counts.values)

# Add numbers above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Histogram of Scores')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()


plt.savefig('mFS_distribution.png', dpi=250)
