import lang2vec.lang2vec as l2v
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Define languages (ISO)
languages = [
    'ace', 'ace', 'acm', 'acq', 'aeb', 'afr', 'ajp', 'aka',
    'amh', 'apc', 'arb', 'arb', 'ars', 'ary', 'arz', 'asm',
    'ast', 'awa', 'ayr', 'azb', 'azj', 'bak', 'bam', 'ban',
    'bel', 'bem', 'ben', 'bho', 'bjn', 'bjn', 'bod', 'bos',
    'bug', 'bul', 'cat', 'ceb', 'ces', 'cjk', 'ckb', 'crh',
    'cym', 'dan', 'deu', 'dik', 'dyu', 'dzo', 'ell', 'eng',
    'epo', 'est', 'eus', 'ewe', 'fao', 'fij', 'fin', 'fon',
    'fra', 'fur', 'fuv', 'gla', 'gle', 'glg', 'grn', 'guj',
    'hat', 'hau', 'heb', 'hin', 'hne', 'hrv', 'hun', 'hye',
    'ibo', 'ilo', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab',
    'kac', 'kam', 'kan', 'kas', 'kas', 'kat', 'knc', 'knc',
    'kaz', 'kbp', 'kea', 'khm', 'kik', 'kin', 'kir', 'kmb',
    'kmr', 'kon', 'kor', 'lao', 'lij', 'lim', 'lin', 'lit',
    'lmo', 'ltg', 'ltz', 'lua', 'lug', 'luo', 'lus', 'lvs',
    'mag', 'mai', 'mal', 'mar', 'min', 'min', 'mkd', 'mlt',
    'mni', 'mos', 'mri', 'mya', 'nld', 'nno', 'nob', 'npi',
    'nso', 'nus', 'nya', 'oci', 'ory', 'pag', 'pan', 'pap',
    'pes', 'pol', 'por', 'prs', 'pbt', 'quy', 'ron', 'run',
    'rus', 'sag', 'san', 'sat', 'scn', 'shn', 'sin', 'slk',
    'slv', 'smo', 'sna', 'snd', 'som', 'sot', 'spa', 'als',
    'srd', 'srp', 'ssw', 'sun', 'swe', 'swh', 'szl', 'tam',
    'tat', 'tel', 'tgk', 'tgl', 'tha', 'tir', 'taq', 'taq',
    'tpi', 'tsn', 'tso', 'tuk', 'tum', 'tur', 'twi', 'tzm',
    'uig', 'ukr', 'umb', 'urd', 'uzn', 'vec', 'vie', 'war',
    'wol', 'xho', 'ydd', 'yor', 'yue', 'zho', 'zho', 'zsm', 'zul'
]

language_subset = [
    'arb', 'fra', 'spa', 'hin',
    'zho', 'eng', 'cym', 'fin',
    'hun', 'zul', 'nld', 'ita',
    'vie', 'swh', 'jpn', 'deu',
    'ind', 'urd', 'rus', 'por',
    'ben'
]

for language in language_subset:
    if language not in languages:
        print(f'Language {language} not in ISO languages')

# Load syntactic features
features = l2v.get_features(languages + ['en'], 'syntax_knn')

# Calculate cosine similarity relative to English
cosine_similarities = {}
for language in language_subset:
    # 1 - cosine distance to make it cosine similarity
    cosine_similarities[language] = 1 - cosine(features['en'], features[language])

sorted_similarities = dict(sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True))

# Print results
print("Cosine Similarity to English:")
for language, similarity in sorted_similarities.items():
    print(f"{language}: {similarity}")

selected_languages_with_scores = [
    ("eng", 1),                # English
    ("fra", 0.8117540630909456),  # French
    ("deu", 0.9025419790150204),  # German
    ("rus", 0.8117540630909456),  # Russian
    ("por", 0.8423970023882502),  # Portuguese
    ("ita", 0.8577941541015601),  # Italian
    ("spa", 0.8215938655409548),  # Spanish
    ("nld", 0.9242930987335743),  # Dutch
    ("fin", 0.7107724707650861),  # Finnish
    ("hun", 0.6939542359352535),  # Hungarian
    ("zho", 0.7107724707650861),  # Chinese
    ("ben", 0.5834482328466172),  # Bengali
    ("hin", 0.6161953991557162),  # Hindi
    ("arb", 0.6443860762255925),  # Arabic
    ("amh", 0.5881716976750462),  # Amharic
    ("zul", 0.44189395817353094), # Zulu
    ("urd", 0.6161953991557162),  # Urdu
    ("swh", 0.47384831384339077), # Swahili
    ("yor", 0.6003002251876642),  # Yoruba
    ("cym", 0.7188946323483355)   # Welsh
]

# Separate the data into labels and values for plotting
labels, values = zip(*selected_languages_with_scores)

# Create a new figure with a specified figure size
plt.figure(figsize=(12, 1))

# Plotting the data as a scatter plot on a line
scatter = plt.scatter(values, [0]*len(values), alpha=0.17, c='blue')

texts = []
# Define initial y position for the first text label
y_pos = 0
# Define the vertical space needed between texts
delta_y = 0.01

# Sort values and labels together based on values
sorted_values_labels = sorted(zip(values, labels))

# Keep track of the previous value to detect overlaps
previous_value = float("-inf")

for value, label in sorted_values_labels:
    # If the value is the same as the previous one, increase y_pos to shift the label up
    if round(value,2) == round(previous_value,2):
        y_pos += delta_y
    else:
        # Reset y_pos if the value is different (no overlap)
        y_pos = 0
    texts.append(plt.text(value, y_pos, label, ha='center', va='bottom', rotation=70, fontsize=7))
    previous_value = value

# Hide the y-axis as it's a 1D plot
plt.gca().axes.get_yaxis().set_visible(False)

# Adding a grid for better readability of x values
plt.grid(True, axis='x', linestyle='--')

# Set a title for the plot
plt.title('Language Similarity Score (lang2vec, syntatic_knn) to English')

# Show the plot
plt.show()