import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import os
os.chdir('multilingual-llm-trustworthiness/research')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_de(df, english_lang='en'):
    """
    Plots the difference in the 'a' metric between each language and English for each model.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'language', 'model', and 'a' columns.
        english_lang (str): The language code for English in the DataFrame (default is 'en').
    """
    # Find the unique models in the DataFrame
    models = df['model'].unique()

    # Assign unique colors
    colors = plt.cm.get_cmap('tab10', len(models))
    model_colors = {model: colors(i) for i, model in enumerate(models)}

    plt.figure(figsize=(12, 8))
    colormap = cm.get_cmap('coolwarm')

    # Bar width and positioning
    bar_width = 0.35
    index = np.arange(len(df['lang'].unique()))

    for i, model in enumerate(models):
        # Filter the DataFrame for the current model
        df_model = df[df['model'] == model]

        # Find the accuracy for English in the current model
        english_accuracy = df_model[df_model['lang'] == english_lang]['a'].values[0]

        # Calculate the difference with English accuracy
        df_model['a_diff'] = df_model['a'] - english_accuracy

        # Sort the DataFrame by the difference
        df_model = df_model.sort_values(by='a_diff')

        # Plot the bars side by side
        plt.bar(index + i * bar_width, df_model['a_diff'], bar_width, label=model,
                color=model_colors[model])

    # Add title and labels
    plt.title('Difference in Metric "a" Compared to English per Model')
    plt.xlabel('Language')
    plt.ylabel('Difference in Metric "a"')

    # Set the x-ticks to the languages
    plt.xticks(index + bar_width / 2 * (len(models) - 1), df_model['lang'], rotation=45)

    # Add a legend
    plt.legend(title='Model')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

df = pd.read_csv('metrics.csv')
plot_de(df)