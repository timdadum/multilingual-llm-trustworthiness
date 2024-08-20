import lang2vec.lang2vec as l2v
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Print the available feature sets to understand the options
print(l2v.available_feature_sets())

# List of languages (ISO 639-3 codes)
languages_iso639_3 = [
    'eng',  # English
    'rus',  # Russian
    'deu',  # German
    'jpn',  # Japanese
    'spa',  # Spanish
    'fra',  # French
    'cmn',  # Chinese (Mandarin)
    'ita',  # Italian
    'por',  # Portuguese
    'nld',  # Dutch
    'vie',  # Vietnamese
    'ind',  # Indonesian
    'ara',  # Arabic
    'swe',  # Swedish
    'hun',  # Hungarian
    'fin',  # Finnish
    'hin',  # Hindi
    'ben',  # Bengali
    'lav',  # Latvian
    'urd',  # Urdu
    'cym',  # Welsh
    'swa',  # Swahili
    'amh',  # Amharic
    'zul',  # Zulu
    'mri'   # Maori
]

# Retrieve different types of features
# geological_vectors = l2v.get_features(languages_iso639_3, 'geo')
# genetic_vectors = l2v.get_features(languages_iso639_3, 'genetic') # Not directly usable in L2V
inventory_vectors = l2v.get_features(languages_iso639_3, 'inventory_average')
syntactic_vectors = l2v.get_features(languages_iso639_3, 'syntax_average')
phonological_vectors = l2v.get_features(languages_iso639_3, 'phonology_average')

def calculate_similarity(u, v, lang):
    """Any '--' elements are disregarded in the distance metric"""
    # Turn lists into arrays
    u = np.array(u, dtype=object)
    v = np.array(v, dtype=object)
    
    # Filter invalid values ('--')
    mask = (u != '--') & (v != '--')
    print(f'{len(mask) - np.sum(mask)} / {len(u)} missing elements for language {lang}')
    
    # With the invalid (string) filtered, turn arrays into numerical arrays
    u = np.array(u[mask])
    v = np.array(v[mask])
    
    # Calculate mean distance and return similarity if u and v both contain features
    if len(u) > 0 and len(v) > 0:
        return 1 - np.mean(np.abs(u-v))
    else:
        return np.nan

# Always calculate distances between English ('eng') and every other language
distances = {'inventory': dict(),
             'syntax': dict(),
             'phonological': dict(),
             'featural': dict()}

for lang_code in languages_iso639_3:
    # Calculate distances between English and the current language
    sinv = calculate_similarity(inventory_vectors['eng'], inventory_vectors[lang_code], lang_code)
    ssyn = calculate_similarity(syntactic_vectors['eng'], syntactic_vectors[lang_code], lang_code)
    spho = calculate_similarity(phonological_vectors['eng'], phonological_vectors[lang_code], lang_code)
    sfea = np.nanmean([sinv, ssyn, spho])  # Aggregate feature distance
    
    # Append the results to the distances dictionary
    distances['inventory'][lang_code] = sinv
    distances['syntax'][lang_code] = ssyn
    distances['phonological'][lang_code] = spho
    distances['featural'][lang_code] = sfea

# Convert the nested distances dictionary to a DataFrame
distance_df = pd.DataFrame(distances)

# Convert the nested distances dictionary to a DataFrame
distance_df = pd.DataFrame(distances)

# Add an index column for language codes
distance_df.index = languages_iso639_3

# Reset the index to have a column for languages
distance_df.reset_index(inplace=True)
distance_df.rename(columns={'index': 'Language'}, inplace=True)

# Color palette for colorblind-friendly visualization
colors = {
    'inventory': '#CC79A7',  # Purple
    'syntax': '#009E73',     # Green
    'phonological': '#D55E00',  # Orange
    'featural': '#0072B2'    # Blue
}

# Sort by featural similarity in ascending order
distance_df.sort_values(by='featural', ascending=False, inplace=True)

# Verify the DataFrame after sorting
print("After sorting by featural similarity:")
print(distance_df[['Language', 'featural']])

def plot_similarity(df):
    n_languages = len(df)
    bar_width = 0.2  # Width of each bar
    gap = 0.3        # Small gap between groups of bars for different languages
    x_positions = [] # To store the center x positions for each language
    x_offset = 0     # Initial x position

    plt.figure(figsize=(14, 4))  # Adjust the size to fit more bars

    # Find the minimum non-zero similarity for y-axis limit
    min_value = df[['inventory', 'syntax', 'phonological', 'featural']].replace(0, np.nan).min().min() - 0.05
    min_value = max(min_value, 0)  # Ensure it doesn't go below 0

    # Plot each feature type side by side, skipping NaN values and applying opacity
    for i, row in df.iterrows():
        bars_plotted = 0
        
        if pd.notna(row['inventory']):
            plt.bar(x_offset, row['inventory'], color=colors['inventory'], edgecolor='white', width=bar_width, label='Inventory Similarity' if i == 0 else "", alpha=0.35)
            x_offset += bar_width
            bars_plotted += 1
        
        if pd.notna(row['syntax']):
            plt.bar(x_offset, row['syntax'], color=colors['syntax'], edgecolor='white', width=bar_width, label='Syntax Similarity' if i == 0 else "", alpha=0.35)
            x_offset += bar_width
            bars_plotted += 1
        
        if pd.notna(row['phonological']):
            plt.bar(x_offset, row['phonological'], color=colors['phonological'], edgecolor='white', width=bar_width, label='Phonological Similarity' if i == 0 else "", alpha=0.35)
            x_offset += bar_width
            bars_plotted += 1
        
        if pd.notna(row['featural']):
            plt.bar(x_offset, row['featural'], color=colors['featural'], edgecolor='white', width=bar_width, label='Featural Similarity' if i == 0 else "")
            x_offset += bar_width
            bars_plotted += 1

        # Store the center x position for this language group
        if bars_plotted > 0:
            x_positions.append(x_offset - (bars_plotted * bar_width) / 2)
            x_offset += gap

    # Add a legend
    plt.legend(loc='upper right', frameon=False, fontsize='small')

    # Titles and labels
    plt.title('Mean feature similarity to English across different linguistic features based on the URIEL database.', fontsize=14)
    plt.xticks(x_positions, df['Language'], rotation=45, ha='right', fontsize=10)
    plt.ylabel('Similarity to English (0-1)', fontsize=12)

    # Set y-limit starting from the minimum non-zero similarity
    plt.ylim(min_value, 1.05)

    # Add a light grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_similarity(distance_df)
# distance_df.to_csv('language_features.csv')