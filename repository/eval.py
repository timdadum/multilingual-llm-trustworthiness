import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

def sort_metrics(df):
    """
    Sorts the DataFrame by each metric in descending order.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'language', 'a', 'b', 'y', and 'k' columns.

    Returns:
        dict: A dictionary containing sorted DataFrames for each metric.
    """
    sorted_metrics = {
        'a': df.sort_values(by=['a'], ascending=False),
        'b': df.sort_values(by=['b'], ascending=False),
        'y': df.sort_values(by=['y'], ascending=False),
        'k': df.sort_values(by=['k'], ascending=False)
    }
    return sorted_metrics

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

def plot_separate_metrics_per_language(df):
    """
    Plots separate bar charts for metrics 'a', 'b', 'y', and 'k' per language.
    Each chart is labeled directly with the language for clarity and colored based on the value.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'language', 'a', 'b', 'y', and 'k' columns.
    """
    # Sort the DataFrame for each metric
    sorted_metrics = sort_metrics(df)

    # Create a colormap
    colormap = cm.get_cmap('viridis')  # You can choose any colormap you prefer

    # Set up the figure
    plt.figure(figsize=(14, 16))  # Adjust the size to fit all plots nicely

    # Plot each metric
    for i, (metric, data) in enumerate(sorted_metrics.items(), 1):
        plot_single_metric(data, metric, colormap, i)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()