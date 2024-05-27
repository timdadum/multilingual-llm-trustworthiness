import pandas as pd
import os

df = pd.read_csv('multilingual-llm-trustworthiness/repository/features/language_features.csv')

print(df.to_string()) 