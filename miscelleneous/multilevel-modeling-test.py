import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Example Data Creation
np.random.seed(42)

# Simulate data for 20 languages, 10 models, 5 benchmarks
languages = [f'Lang_{i}' for i in range(1, 21)]
models = [f'Model_{i}' for i in range(1, 11)]
benchmarks = [f'Benchmark_{i}' for i in range(1, 6)]

# Create a DataFrame with one row per (Language, Model, Benchmark)
data = pd.DataFrame(
    [(lang, mod, bench) for lang in languages for mod in models for bench in benchmarks],
    columns=['Language', 'Model', 'Benchmark']
)

# Simulate linguistic, model, and benchmark features
data['Linguistic_Feature'] = np.random.randn(len(data))
data['Model_Feature'] = np.random.randn(len(data))
data['Benchmark_Feature'] = np.random.randn(len(data))

# Simulate random effects and observed risk scores (averaged over 10,000 samples)
language_effects = np.random.randn(len(languages))
model_effects = np.random.randn(len(models))
benchmark_effects = np.random.randn(len(benchmarks))

# Assign random effects for each group
data['Lang_Effect'] = data['Language'].map(dict(zip(languages, language_effects)))
data['Model_Effect'] = data['Model'].map(dict(zip(models, model_effects)))
data['Bench_Effect'] = data['Benchmark'].map(dict(zip(benchmarks, benchmark_effects)))

# Generate observed risk score (R) based on fixed and random effects
data['Risk_Score'] = (0.5 * data['Linguistic_Feature'] +
                      0.3 * data['Model_Feature'] +
                      0.2 * data['Benchmark_Feature'] +
                      data['Lang_Effect'] + 
                      data['Model_Effect'] + 
                      data['Bench_Effect'] +
                      np.random.normal(scale=0.1, size=len(data)))  # Add small noise

# Model the Risk_Score using a hierarchical structure with random effects for Language, Model, and Benchmark

# MixedLM formula: fixed effects + random intercepts for Language, Model, Benchmark
model_formula = "Risk_Score ~ Linguistic_Feature + Model_Feature + Benchmark_Feature"
# Fit the random effects for Language, Model, and Benchmark
mixed_model = mixedlm(model_formula, data, groups=data['Language'],
                      re_formula="~Model + Benchmark",  # Random effects for both Model and Benchmark
                      vc_formula={"Model": "0 + Model", "Benchmark": "0 + Benchmark"})  # Allow each Model and Benchmark to have its own random effect

# Fit the model
fit = mixed_model.fit()

# Print the summary
print(fit.summary())