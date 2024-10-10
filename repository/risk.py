### MAIN FILE FOR CONFORMAL RISK CONTROL ###

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def conformal_risk_control(benchmark, confidence_intervals: list):
    """
    This function performs conformal risk control (CRC) by evaluating the provided benchmark
    against a set of confidence intervals.

    Args:
        benchmark: MultilingualBenchmark object to apply CRC to
        confidence_intervals (list): A list of confidence intervals on which finite-sample risk guarantees
                                     are calculated. Confidence intervals should be upper bound confidence values, e.g. p95 -> 0.95
    
    Returns:
        risks: A nested dictionary with calculated risks for different confidence levels:
            {
                "nl": {
                    "ncr95": 0.124,   # 95% confidence upper bound nonconformality score
                    "ncr98": 0.263,   # 98% confidence upper bound nonconformality score
                    "ncr99": 0.541,   # 99% confidence upper bound nonconformality score
                    "mc95": 0.124,    # 95% confidence upper bound miscoverage rate
                    "mc98": 0.263,    # 98% confidence upper bound miscoverage rate
                    "mc99": 0.541,    # 99% confidence upper bound miscoverage rate
                    "conf": 0.056     # Fraction of conformal responses
                }
            }
    """
    risks = {}
    thresholds_per_language = {}

    for lang in benchmark.languages:
        # Get nonconformity scores for the language
        nonconformity_scores = get_nonconformity_scores(benchmark.samples, lang)

        # Split the nonconformity scores into calibration and prediction sets
        calibration_set, prediction_set = split(nonconformity_scores)

        # Calculate nonconformity thresholds for each confidence level
        thresholds = {f'ncr{int(confidence * 100)}': calculate_threshold(calibration_set, lang, confidence) 
                      for confidence in confidence_intervals}

        # Store thresholds for plotting
        thresholds_per_language[lang] = thresholds

        # Calculate upper bound nonconformity score and miscoverage rate for prediction set
        upper_bounds = {f'mc{int(confidence * 100)}': calculate_upper_bound_nonconformality(prediction_set, thresholds[f'ncr{int(confidence * 100)}']) 
                        for confidence in confidence_intervals}

        # Calculate the conformity score (fraction of perfect responses)
        conformity_score = calculate_conformity(nonconformity_scores)

        # Combine results for this language
        risks[lang] = {**thresholds, **upper_bounds, "conf": conformity_score}

        # Plot calibration set and thresholds for this language
        plot_calibration_with_thresholds(calibration_set, thresholds, confidence_intervals, lang)

    return risks

def calculate_conformity(nonconformity_scores):
    """
    Calculate the fraction of samples with perfect conformity (no errors - nonconformity level of 0).
    
    Args:
        nonconformity_scores: List of nonconformity scores.
    
    Returns:
        Fraction of samples with perfect conformity.
    """
    perfect_responses = nonconformity_scores.count(0)
    return perfect_responses / len(nonconformity_scores)

def calculate_nonconformity(evaluations):
    """
    Takes a list of binary evaluations, returns the nonconformity score
    
    Args:;
        evalutions: List of floats (1 or 0) representing LLM output evaluations
    Returns:
        score: Float value of nonconformity 
    
    Current nonconformity function: f(X_n) = SUM(X_n != X_true) / N
    """
    return evaluations.count(0) / len(evaluations)

def get_nonconformity_scores(samples, lang):
    """
    Get nonconformity scores for each sample in the dataset for the specified language.
    
    Args:
        samples: List of dictionaries containing evaluation data for each sample.
        lang: Language code for which evaluations are calculated.
    
    Returns:
        List of nonconformity scores.
    """
    scores = []
    for sample in samples:
        scores.append(calculate_nonconformity(sample.evaluations[lang]))
    return scores

def calculate_threshold(calibration_set, lang, confidence_level=0.95):
    """
    Calculates the nonconformity threshold based on the given confidence level.

    Args:
        calibration_set: List of nonconformity scores from the calibration set.
        confidence_level: Confidence level (e.g. 0.95 for 95%).

    Returns:
        Nonconformity threshold at the specified confidence level.
    """
    percentile = (1 - confidence_level) * 100  # For 95%, take the 5th percentile
    threshold = np.percentile(calibration_set, percentile)
    
    return threshold

def plot_calibration_with_thresholds(calibration_set, thresholds, confidence_intervals, lang):
    """
    Plots the calibration set distribution along with threshold lines for different confidence intervals.

    Args:
        calibration_set: List of nonconformity scores from calibration set.
        thresholds: Dictionary of thresholds for different confidence intervals.
        confidence_intervals: List of confidence intervals used.
        lang: Language being processed.
    """
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    plt.figure(figsize=(8, 6))
    
    # Plot the calibration set distribution
    plt.hist(calibration_set, bins=30, alpha=0.7, color='gray', label='Calibration Set')
    
    # Plot a dashed line for each threshold
    for i, confidence in enumerate(confidence_intervals):
        threshold = thresholds[f'ncr{int(confidence * 100)}']
        plt.axvline(threshold, color=colors[i % len(colors)], linestyle='--', linewidth=2, label=f'{int(confidence * 100)}% Threshold ({threshold:.2f})')

    plt.title(f'Distribution of nonconformity scores (calibration set) for {lang}')
    plt.xlabel('Nonconformity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def split(scores, calibration_size=0.2):
    """
    Split nonconformity scores into a calibration set and a prediction set.
    
    Args:
        scores: List of nonconformity scores.
        calibration_size: Proportion of the dataset to be used for calibration.
    
    Returns:
        calibration_set: The calibration set.
        prediction_set: The prediction set.
    """
    calibration_set, prediction_set = train_test_split(scores, test_size=calibration_size)
    return calibration_set, prediction_set

def calculate_upper_bound_nonconformality(prediction_set, threshold):
    """
    Calculate the upper bound nonconformity score for the prediction set given the threshold.
    
    Args:
        prediction_set: List of nonconformity scores in the prediction set.
        threshold: The nonconformity threshold determined from the calibration set.
    
    Returns:
        The proportion of the prediction set that exceeds the threshold (upper bound).
    """
    exceed_count = sum(score > threshold for score in prediction_set)
    return exceed_count / len(prediction_set)