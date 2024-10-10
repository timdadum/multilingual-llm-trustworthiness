import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

# def sort_metrics(df):
#     """
#     Sorts the DataFrame by each metric in descending order.

#     Parameters:
#         df (pd.DataFrame): DataFrame containing 'language', 'a', 'b', 'y', and 'k' columns.

#     Returns:
#         dict: A dictionary containing sorted DataFrames for each metric.
#     """
#     sorted_metrics = {
#         'a': df.sort_values(by=['a'], ascending=False),
#         'b': df.sort_values(by=['b'], ascending=False),
#         'y': df.sort_values(by=['y'], ascending=False),
#         'k': df.sort_values(by=['k'], ascending=False)
#     }
#     return sorted_metrics

def plot_single_metric(data, metric, colormap, subplot_index):
    """
    Plots a single metric as a bar chart with colors based on relative values.

    Parameters:
        data (pd.DataFrame): The sorted DataFrame for the specific metric.
        metric (str): The metric to be plotted ('a', 'b', 'y', or 'k').
        colormap: The colormap used to assign colors based on value.
        subplot_index (int): Index of the subplot.
    """
    plt.subplot(4, 1, subplot_index)

    # Normalize the values to be in the range [0, 1] for color mapping
    normalized_values = (data[metric] - data[metric].min()) / (data[metric].max() - data[metric].min())

    # Plot bars with colors based on the normalized values
    bar_width = 0.7
    indices = np.arange(len(data))
    bar_colors = colormap(normalized_values)
    plt.bar(indices, data[metric], bar_width, color=bar_colors)

    # Add text annotations with language names directly on top of bars
    for i, value in enumerate(data[metric]):
        plt.text(i, value + 0.01, data['lang'].iloc[i], ha='center', va='bottom', fontsize=10)

    # Set labels and titles
    plt.title(f'{metric.upper()}', fontsize=14)
    plt.xticks(indices, data['lang'], rotation=45, ha='right', fontsize=10)

    # Grid for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.6)

def plot(df):
    """
    Plots an accuracy bar chart per language.

    Parameters:
        df (pd.DataFrame): DataFrame containing language and accuracy. 
                           The DataFrame is sorted by accuracy before plotting.
    """
    # Sort the DataFrame by accuracy column
    df = df.sort_values(by='accuracy')

    # Create a colormap
    colormap = cm.get_cmap('viridis')

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Plot the 'a' metric versus language
    bars = plt.bar(df['lang'], df['accuracy'], color=colormap(df['accuracy'] / df['accuracy'].max()))
    plt.title('Accuracy ')
    plt.xlabel('Language')
    plt.ylabel('Accuracy')

    # Label each bar with the corresponding value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_de(df, english_lang='en'):
    """
    Plots the difference in the accuracy between each language and English.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'language' and 'accuracy' columns.
        english_lang (str): The language code for English in the DataFrame (default is 'en').
    """
    # Find the accuracy for English
    english_accuracy = df[df['lang'] == english_lang]['accuracy'].values[0]

    # Calculate the difference with English accuracy
    df['accuracy_diff'] = df['accuracy'] - english_accuracy

    # Sort the DataFrame by the difference
    df = df.sort_values(by='accuracy_diff')

    # Create a colormap
    colormap = cm.get_cmap('coolwarm')

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Plot the difference in accuracy versus language
    bars = plt.bar(df['lang'], df['accuracy_diff'], color=colormap(df['accuracy_diff'] / df['accuracy_diff'].abs().max()))
    plt.title('Relative accuracy difference Compared to English')
    plt.xlabel('Language')
    plt.ylabel('Difference in accuracy')

    # Label each bar with the corresponding value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()