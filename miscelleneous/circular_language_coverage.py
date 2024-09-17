import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('commoncrawl_language_stats.csv')

def normalize_data(data):
    # Find the minimum and maximum values
    min_val = min(data)
    max_val = max(data)
    
    # Apply min-max normalization
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized

# Extract data
languages = data['language'].tolist()
feature1 = data['percentage'].tolist()  # Assuming 'percentage' is one feature
feature2 = data['population'].tolist()  # Assuming 'population' is another feature

# Normalize
feature1 = normalize_data(feature1)
feature2 = normalize_data(feature2)

# Number of variables
num_vars = len(languages)

# Create angles for the polar plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# Complete the loop
angles += angles[:1]
feature1 += feature1[:1]  # Ensure the graph connects back to the start
feature2 += feature2[:1]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.fill(angles, feature1, color='red', alpha=0.25, label='Percentage')
ax.fill(angles, feature2, color='blue', alpha=0.25, label='Population')

# Labels for each language
ax.set_xticks(angles[:-1])
ax.set_xticklabels(languages, size=8)

# Adding a legend
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
