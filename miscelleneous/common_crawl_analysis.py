import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from CSV
print(os.getcwd())
data = pd.read_csv('multilingual-llm-trustworthiness\miscelleneous\language_features.csv')

# Normalize data function to scale from 1 to 100
def normalize_data(data):
    return ((data - data.min()) / (data.max() - data.min()) * 99) + 1

# Apply logarithmic transformation (base 10) to the normalized data
data['log_percentage'] = np.log10(data['percentage'])
data['log_population'] = np.log10(data['population(100m)'])

# Sort data by log_normalized_population for radial ordering
data = data.sort_values(by='log_population', ascending=False)

# Extracting sorted data for plotting
languages = data['language'].tolist()
log_percentages = data['log_percentage'].tolist()
log_populations = data['log_population'].tolist()

# Number of variables
num_vars = len(languages)

print(data)

# Create angles for the polar plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Ensure the graph connects back to the start
log_percentages += log_percentages[:1]
log_populations += log_populations[:1]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.fill(angles, log_percentages, color='orange', alpha=0.5, label='Percentage of language data in CC-MAIN-2023-50 (log)')
ax.fill(angles, log_populations, color='violet', alpha=0.4, label='Speakers (in 100m, log)')

# Adding less opaque lines for specific values
for value in [-2.75, -2.5, -2.25, -1.75, -1.5, -1.25, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.25, 1.5]:
    ax.plot(angles, [value] * len(angles), linestyle='dashed', linewidth=1, alpha=0.15, color='grey')

# Labels for each language
ax.set_xticks(angles[:-1])
ax.set_xticklabels(languages, size=8)

# Adding a legend
ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.1), prop = { "size": 6 })

# Adjust the y-axis grid lines
# We use log scale values for ticks since the values are log-transformed
ticks = [-3, -2, -1, 0, 1]
ax.set_yticks(ticks)
ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=6)  # Modify here for different sizes

# Set the regular grid lines opacity
# ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.25)  # Setting the opacity here

plt.show()
