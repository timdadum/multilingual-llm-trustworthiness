import pandas as pd
import matplotlib.pyplot as plt
import os

filename = "../Thesis/multilingual-train-deduplicated.csv"
column_name = "lang" # language

df = pd.read_csv(filename)

print(df.head)
