import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata
import textwrap
from logger import logger

# from evaluate import load
# bertscore = load("bertscore")
# preds = ["The planet Jupiter", "You would be bored"]
# refs = ["Jupiter", "There would be nothing you could do"]
# results = bertscore.compute(predictions=preds,
#                             references=refs,
#                             lang='en')
# print(results)




def plot_accuracy_vs_sim(df, poly_order=3):
    """Takes a dataframe with results, plots accuracy versus language similarity to English (scatter) /w polynomial regression
    
    poly_order = the polynomial order of the fitted polynomial regression"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(x='sim', y='accuracy', data=df, s=50)

    # Plot language labels
    for i, row in df.iterrows():
        plt.text(row['sim'], row['accuracy'], row['language'], fontsize=9, ha='right')
    
    # Plot polynomial regression
    x = df['sim'].values
    y = df['accuracy'].values
    
    try:
        coeffs = np.polyfit(x, y, poly_order)
        poly_eq = np.poly1d(coeffs)
        
        # Generate x values for plotting the polynomial line
        x_poly = np.linspace(x.min(), x.max(), 100)
        y_poly = poly_eq(x_poly)
        
        # Plot the polynomial regression line
        plt.plot(x_poly, y_poly, color='red', linestyle='--', label=f'Polynomial Regression (order {poly_order})')
    except Exception as e:
        logger.error("An error occurred during polynomial fitting:", e)

    title = 'Scatter plot of per-language experiment accuracy (internal knowledge - simple QA) vs. language similarity to English as measured in cosine similarity between (lang2vec) language vectors'
    wrapped_title = textwrap.fill(title, width=80)
    plt.title(wrapped_title)
    
    plt.xlabel('Language vector similarity (cosine) to English language vector')
    plt.ylabel('Accuracy on QA benchmark (%)')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_accuracy_vs_percentage(df):
    """Takes a dataframe with results, plots accuracy versus CC Percentage (scatter)"""
    
    plt.figure(figsize=(10, 6))
    
    for i, row in df.iterrows():
        plt.text(row['percentage'], row['accuracy'], row['language'], fontsize=9, ha='right')
    
    sns.scatterplot(x='percentage', y='accuracy', data=df, s=100)
    
    plt.title('Scatter plot of per-language accuracy (GPT-3.5) vs. web data availability (percentage in Common Crawl*)')
    plt.xlabel(r'Web data availability (% in Common Crawl)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_3d_accuracy_vs_sim_and_percentage(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['sim'], df['percentage'], df['accuracy'], c='b', marker='o')
    ax.set_title('3D Scatter plot of language accuracy (GPT-3.5) vs. similarity to English (lang2vec, cosine) and web data availability (percentage in Common Crawl*)')
    ax.set_xlabel('Similarity to English (cosine similarity based on lang2vec "syntax_knn" features)')
    ax.set_ylabel(r'Web data availability (% in Common Crawl)')
    ax.set_zlabel('Accuracy (%)')
    
    caption = (
        'Figure 3: 3D Scatter plot of per-language experiment accuracy (internal knowledge - simple QA) over language similarity to English and web data availability. '
        'The found accuracies have been obtained with small subset of n=100 QA pairs, identical between languages. Translation of QA pairs is done through Google Translate, which is particularly noisy for low-resource languages. '
        'Used model is GPT-3.5-turbo, both for querying the question as well as for evaluation.'
    )
    wrapped_caption = textwrap.fill(caption, width=260)
    plt.legend()
    plt.figtext(0.5, 0.01, wrapped_caption, ha='center', fontsize=10)
    
    plt.show()

def plot_surface_accuracy_vs_sim_and_percentage(df):
    """Takes a dataframe with results, plots accuracy versus language similarity to English and CC Percentage (3D scatter)"""
    plt.figure(figsize=(10, 6))
    df = df.dropna(subset=['sim', 'percentage', 'accuracy'])
    
    x = df['sim']
    y = df['percentage']
    z = df['accuracy']
    
    # Increase the resolution of the grid
    xi, yi = np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200)
    xi, yi = np.meshgrid(xi, yi)
    
    # Use cubic interpolation for a smoother contour plot
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Mask NaN values to avoid gaps in the plot
    zi = np.ma.masked_where(np.isnan(zi), zi)
    
    # Plot the filled contour
    contour = plt.contourf(xi, yi, zi, levels=256, cmap='viridis')
    
    ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Add a color bar
    cbar = plt.colorbar(contour, ticks=ticks)
    cbar.set_label('Accuracy (%)')
    
    # Plot the data points and their labels
    for i, row in df.iterrows():
        plt.text(row['sim'], row['percentage'], row['language'], fontsize=9, ha='right', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    title = '3D surface plot of language accuracy (GPT-3.5) vs. similarity to English (lang2vec, cosine) and web data availability (percentage in Common Crawl*)'
    wrapped_title = textwrap.fill(title, width=80)
    plt.title(wrapped_title)
    
    plt.xlabel('Similarity to English (cosine similarity based on lang2vec "syntax_knn" features)')
    plt.ylabel('Web Data Availability (Percentage in Common Crawl)')
    plt.grid(True)
    
    # Add margins to the plot
    plt.ylim(-0.5, 6.25)
    plt.xlim(0.4, 0.95)
    
    plt.show()


def _extract_accuracies(data: list, languages: list):
    """
    Converts experiment results to a dictionary of accuracies.

    Args:
        data (list of dictionaries): Object containing experiment results in standard format
        languages (list of str): List of languages in data
    Returns:
        scores (dict): A dictionary with per-language accuracy
    """
    scores = {}
    for language in languages:
        try:
            # Filter out null values and count the correct evaluations
            correct = sum(1 for sample in data if '1' in sample.get(f"Evaluation_{language}") and "null" not in sample.get(f"Evaluation_{language}").lower())
            total = sum(1 for sample in data if "null" not in sample.get(f"Evaluation_{language}").lower())
            scores[language] = correct / total if total > 0 else None
        except Exception as e:
            logger.warning(f"No accuracy found for language {language}, outputting None: ({e})")
            scores[language] = None
    return scores
