import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
import os

# Data for main criteria and subcriteria
main_criteria = {
    "Truthfulness": 0.554,
    "Safety": 0.247,
    "Fairness": 0.195,
    "Robustness": 0.386,
    "Privacy": 0.259,
    "Ethics": 0.146
}

subcriteria = {
    "Internal Knowledge (Acc)": 0.576,
    "External Knowledge (Macro F-1)": 0.315,
    "Sycophancy (pref.) (Δ)": 0.685,
    "Adversarial Factuality (Δ)": 0.638,
    "Jailbreak (RtA)": 0.188,
    "Toxicity (Avg)": 0.351,
    "Misuse (RtA)": 0.198,
    "Exaggerated Safety (RtA)": 0.249,
    "Stereotypes (Agree%)": 0.058,
    "Disparagement (p-value)": 0.081,
    "Preference in subjective choice (RtA)": 0.445,
    "Adversarial Robustness (Acc)": 0.310,
    "Out-Of-Distribution Recognition (RtA)": 0.461,
    "Privacy Awareness (normal) (RtA)": 0.326,
    "Privacy Awareness (augmented) (RtA)": 0.002,
    "Privacy Leakage (5-shot) (Acc)": 0.449,
    "Implicit Ethics (Acc)": 0.351,
    "Explicit Ethics (Acc)": 0.004,
    "Emotional Ethics (Acc)": 0.082
}

proprietary_vs_open = {
    "Truthfulness": [0.554, 0.550],
    "Safety": [0.247, 0.251],
    "Fairness": [0.195, 0.365],
    "Robustness": [0.386, 0.488],
    "Privacy": [0.259, 0.179],
    "Ethics": [0.146, 0.293]
}

cwd = os.getcwd()

# Construct the full file path by joining the CWD with the desired filename
main_criteria_path = os.path.join(cwd, "main_criteria_risk_evaluations.pdf")
subcriteria_path = os.path.join(cwd, "subcriteria_risk_evaluations.pdf")
proprietary_vs_open_path = os.path.join(cwd, "proprietary_vs_open_source_comparison.pdf")


# Sorting the data
main_criteria = dict(sorted(main_criteria.items(), key=lambda item: item[1]))
subcriteria = dict(sorted(subcriteria.items(), key=lambda item: item[1]))

# Inverting the color map (green for 0, red for 1)
cmap_inverted = mcolors.LinearSegmentedColormap.from_list("gr", ["green", "yellow", "red"], N=100)

# Plotting Main Criteria
plt.figure(figsize=(10, 2))
main_values = list(main_criteria.values())
main_labels = list(main_criteria.keys())

sns.barplot(x=main_values, y=main_labels, palette=[cmap_inverted(value) for value in main_values])
plt.xlabel('Normalized Risk evaluation')
# plt.title('Risk evaluations per main criterium, averaged over proprietary, open-source and gpt-only evaluations')
plt.grid(True, axis='x', linestyle='--', linewidth=0.7)
plt.xticks(ticks=[i * 0.05 for i in range(14)])  # Increase tick density

# plt.savefig(main_criteria_path, format='pdf')

# Create a ScalarMappable object for the color bar
norm = plt.Normalize(0, 1)  # Assuming the risk evaluation is between 0 and 1
sm = plt.cm.ScalarMappable(cmap=cmap_inverted, norm=norm)
sm.set_array([])  # Required for ScalarMappable to work with colorbar

# Add the color bar to map the color scale to the risk evaluation
cbar = plt.colorbar(sm)
cbar.set_label('Risk evaluation')
plt.gca().invert_yaxis()

# Plotting Subcriteria
plt.figure(figsize=(10, 8))
sub_values = list(subcriteria.values())
sub_labels = list(subcriteria.keys())

sns.barplot(x=sub_values, y=sub_labels, palette=[cmap_inverted(value) for value in sub_values])

plt.xticks(ticks=[i * 0.05 for i in range(14)])  # Increase tick density
plt.xlabel('Normalized Risk evaluation')
# plt.title('Risk evaluations per subcriterium, averaged over proprietary, open-source and gpt-only evaluations')
plt.grid(True, axis='x', linestyle='--', linewidth=0.7)
plt.gca().invert_yaxis()

# plt.savefig(subcriteria_path, format='pdf')

# Create a ScalarMappable object for the color bar
norm = plt.Normalize(0, 1)  # Assuming the risk evaluation is between 0 and 1
sm = plt.cm.ScalarMappable(cmap=cmap_inverted, norm=norm)
sm.set_array([])  # Required for ScalarMappable to work with colorbar

# Add the color bar to map the color scale to the risk evaluation
cbar = plt.colorbar(sm)
cbar.set_label('Risk evaluation')

# Flattening the data for proprietary vs open-source comparison for grouped plot
df_proprietary_open = pd.DataFrame(proprietary_vs_open, index=["Proprietary", "Open-Source"]).T
df_proprietary_open = df_proprietary_open.sort_values(by="Proprietary")

# Create a DataFrame for melted data
df_proprietary_open_melted = df_proprietary_open.reset_index().melt(id_vars="index", var_name="Type", value_name="Risk")

# Create the figure for proprietary vs open-source with grouped bars
plt.figure(figsize=(10, 5))

# Set up the bar width and margin between groups
bar_width = 0.35
bar_spacing = 0.5  # Spacing between groups
positions = []
y_ticks = []

# Iterate over criteria and plot bars for proprietary and open-source grouped together
for i, criterion in enumerate(df_proprietary_open['Proprietary'].index):
    proprietary_value = df_proprietary_open.loc[criterion, 'Proprietary']
    open_source_value = df_proprietary_open.loc[criterion, 'Open-Source']

    # Calculate positions for proprietary and open-source bars
    pos_proprietary = i * (bar_width * 2 + bar_spacing)
    pos_open_source = pos_proprietary + bar_width

    # Store positions for y-ticks
    positions.append((pos_proprietary + pos_open_source) / 2)
    y_ticks.append(criterion)

    # Plot Proprietary with label in the format "Criterion (Proprietary)"
    plt.barh(pos_proprietary, proprietary_value, bar_width, color=cmap_inverted(proprietary_value),
             label='Proprietary' if i == 0 else "", align='center')
    plt.text(proprietary_value + 0.01, pos_proprietary, f"Proprietary", va='center', size='small')

    # Plot Open-Source with label in the format "Criterion (Open-Source)"
    plt.barh(pos_open_source, open_source_value, bar_width, color=cmap_inverted(open_source_value),
             label='Open-Source' if i == 0 else "", align='center')
    plt.text(open_source_value + 0.01, pos_open_source, f"Open-Source", va='center', size='small')

# Set y-ticks at the middle of the grouped bars
plt.yticks(positions, y_ticks)

# Add grid lines to the x-axis
plt.grid(True, axis='x', linestyle='--', linewidth=0.7)

# Create a ScalarMappable object for the color bar
norm = plt.Normalize(0, 1)  # Assuming the risk evaluation is between 0 and 1
sm = plt.cm.ScalarMappable(cmap=cmap_inverted, norm=norm)
sm.set_array([])  # Required for ScalarMappable to work with colorbar

# Add the color bar to map the color scale to the risk evaluation
cbar = plt.colorbar(sm)
cbar.set_label('Risk evaluation')

# Adjust the number of ticks for more grid lines
plt.xticks(ticks=[i * 0.05 for i in range(14)])  # Increase tick density

# Labels and title
plt.xlabel('Normalized Risk evaluation')
plt.ylabel('Criteria')
# plt.title('A comparison of criteria-level risk evaluations between Proprietary vs Open-Source models')

# Layout adjustment
plt.tight_layout()

# Remove the legend
plt.legend([], [], frameon=False)

# Show the plot
plt.show()

# plt.savefig(proprietary_vs_open_path, format='pdf')