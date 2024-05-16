import pandas as pd
import matplotlib.pyplot as plt
import os

filename = "../Thesis/multilingual-train-deduplicated.csv"
column_name = "lang" # language

df = pd.read_csv(filename)

counts = df[column_name].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel(f"Values in '{column_name}")
plt.ylabel("Frequency")
plt.title("Frequency of languages in current dataset")
plt.show()

print("Descriptive statistics for", column_name)
print(df[column_name].describe())

