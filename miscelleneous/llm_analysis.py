import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
filename = "multilingual-llm-trustworthiness/miscelleneous/llms.csv"
df = pd.read_csv(filename, delimiter=';')

# Keep only rows where 'Corpus size (billion tokens)' is not empty or 'Unknown'
df = df[~df['Corpus size (billion tokens)'].isin(['', 'Unknown'])]

# Ensure proper data types
df['Name'] = df['Name'].astype(str)
df['Release date'] = pd.to_datetime(df['Release date'], format='%B %Y')
df['Number of parameters (million)'] = df['Number of parameters (million)'].astype(float)
df['Corpus size (billion tokens)'] = df['Corpus size (billion tokens)'].astype(float)

# Define the list of notable models
notable_models = [
    'GPT-1',
    'BERT',
    'GPT-2',
    'GPT-3',
    'Claude',
    'GLaM',
    'Chinchilla',
    'PaLM',
    'BLOOM',
    'LLaMA',
    'GPT-4',
    'PaLM 2',
    'Llama 2',
    'Mistral 7B',
    'Gemini 1.0',
    'Gemini 1.5',
    'Claude 3',
    'Llama 3.1'
]

# Plotting
plt.scatter(df['Release date'], df['Number of parameters (million)'], color='orange', s=20, label='Other Models')

# Highlight and annotate notable models
for i in range(len(df)):
    if df['Name'].iloc[i] in notable_models:
        plt.scatter(df['Release date'].iloc[i], df['Number of parameters (million)'].iloc[i], color='green', s=50, label='Notable Models')
        plt.text(x=df['Release date'].iloc[i],
                 y=df['Number of parameters (million)'].iloc[i],
                 s=df['Name'].iloc[i],
                 fontsize='large', # Increase the font size
                 rotation=15, # Rotate the text
                 ha='left', # Horizontal alignment
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')) # Add text background

# General plot formatting
plt.xlabel('Release Date')
plt.ylabel('Number of parameters (million) of landmark models over time')
plt.yscale('log')
plt.title('Landmark model size over time')
plt.xticks(rotation=45) # Rotate date labels for better readability
plt.tight_layout() # Adjust layout to make room for the rotated date labels
    
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Set y range
plt.ylim([100, 1e6])

# Regression
df['date_counter'] = (df['Release date'] - pd.to_datetime('2018-01-01')).dt.days
X = np.log1p(df[['date_counter']])
y = np.log(df['Number of parameters (million)'])
model = LinearRegression()
model.fit(X, y)

# Count 6 years after 2018, corresponding to the days
counter_X = np.arange(0, 6 * 365, 30)  # Approximate each month by 30 days
log_counter_X = np.log1p(counter_X)
months_to_predict = [pd.to_datetime('2018-01-01') + pd.DateOffset(days=int(days)) for days in counter_X]
y_pred = model.predict(log_counter_X.reshape(-1, 1))

# Plot the regression line
plt.plot(months_to_predict, np.exp(y_pred), label='Logarithmic Regression', color='purple', linestyle='--', alpha=0.25)

# Avoid duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize="10")

plt.show()
