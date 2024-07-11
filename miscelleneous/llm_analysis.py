import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# Load the dataset
filename = "multilingual-llm-trustworthiness\miscelleneous\llms.csv"
df = pd.read_csv(filename, delimiter=';')

# Keep only rows where 'Corpus size (billion tokens)' is not empty or 'Unknown'
df = df[~df['Corpus size (billion tokens)'].isin(['', 'Unknown'])]

# Ensure proper data types
df['Name'] = df['Name'].astype(str)
df['Release date'] = pd.to_datetime(df['Release date'], format='%B %Y')
df['Number of parameters (million)'] = df['Number of parameters (million)'].astype(float)
df['Corpus size (billion tokens)'] = df['Corpus size (billion tokens)'].astype(float)

# Plotting
plt.scatter(df['Release date'], df['Number of parameters (million)'], color='orange')
plt.xlabel('Release Date')
plt.ylabel('Model size (million parameters) of landmark models over time')
plt.yscale('log')
plt.title('Corpus size (billion tokens) of landmark models over time')
plt.xticks(rotation=45) # Rotate date labels for better readability
plt.tight_layout() # Adjust layout to make room for the rotated date labels
    
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Set y range
plt.ylim([80, 5e6])   

# Plot point names
for i in range(len(df)):
    plt.text(x=df['Release date'].iloc[i],
             y=df['Number of parameters (million)'].iloc[i],
             s=df['Name'].iloc[i],
             fontsize='xx-small', # Increase the font size
             rotation=15, # Rotate the text
             ha='left', # Horizontal alignment
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')) # Add text background


# Regression
df['date_counter'] = (df['Release date'] - pd.to_datetime('2018-01-01')).dt.days
X = np.log1p(df[['date_counter']])
y = np.log(df['Number of parameters (million)'])
model = LinearRegression()
model.fit(X, y)

# Count 6 years after 2018, corresponding to the days
counter_X = np.arange(0, 6 * 365, 30)  # Approximate each month by 30 days
log_counter_X = np.log1p(counter_X)
months_to_predict = [pd.to_datetime('2018-01-01') + pd.DateOffset(days=int(days)) for days in counter_X]
y_pred = model.predict(log_counter_X.reshape(-1, 1))

plt.plot(months_to_predict, np.exp(y_pred), label='Logarithmic Regression', color='purple', linestyle='--', alpha=0.25)
plt.legend(loc="upper left", fontsize="10")

plt.show()
