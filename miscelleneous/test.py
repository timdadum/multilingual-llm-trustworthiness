import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#\ generate dummy data 
np.random.seed(42)

# Assume 5 models, 3 benchmarks, 4 languages
n_models = 5
n_benchmarks = 3
n_languages = 4

# Generate random scores
scores = np.random.rand(n_models, n_benchmarks, n_languages)

# Generate some dummy features for models and languages
model_features = np.random.rand(n_models, 5)
language_features = np.random.rand(n_languages, 3)

data = []

for i in range(n_models):
    for j in range(n_benchmarks):
        english_score = scores[i, j, 0]  # Assume index 0 corresponds to English
        max_abs_diff = np.max(np.abs(scores[i, j, :] - english_score))
        for k in range(n_languages):
            y = (scores[i, j, k] - english_score) / max_abs_diff
            combined_features = np.concatenate((model_features[i], language_features[k]))
            data.append(np.concatenate(([y], combined_features)))

data = np.array(data)
df = pd.DataFrame(data, columns=['y'] + [f'feat_{i}' for i in range(data.shape[1] - 1)])

# Calculate correlations
original_corr = df.corr()

# Fit a model
X = df.drop(columns=['y'])
y = df['y']

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
original_mse = mean_squared_error(y, predictions)

# Verification after the shift and scaling operation
transformed_corr = df.corr()

# Fit a model again to check MSE remains similar
transformed_model = LinearRegression()
transformed_model.fit(X, y)
transformed_predictions = transformed_model.predict(X)
transformed_mse = mean_squared_error(y, transformed_predictions)

# Display results
print("Original Correlation Matrix:\n", original_corr)
print("\nTransformed Correlation Matrix:\n", transformed_corr)
print("\nOriginal MSE:", original_mse)
print("Transformed MSE:", transformed_mse)
