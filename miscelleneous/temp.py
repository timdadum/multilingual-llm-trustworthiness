import pandas as pd

# Load the TSV file into a DataFrame with handling for bad lines
df = pd.read_csv('x-fact.tsv', sep='\t', on_bad_lines='skip')

# Function to display samples per language
def display_samples_per_language(df, num_samples=10):
    # Get the unique languages
    print(df['language'].unique())
    print(df['label'].unique())
    languages = ['de', 'it', 'en']  # You can use df['language'].unique() if needed
    
    for language in languages:
        # Filter the DataFrame for the current language
        language_df = df[df['language'] == language]

        # Reduce samples to unambiguous labels
        reduced_df = language_df[language_df['label'].isin(['true', 'false', 'partly true/misleading'])]

        # Check if there are fewer samples than num_samples, show all
        num_samples_to_show = min(num_samples, len(reduced_df))
        
        if num_samples_to_show == 0:
            print(f"No samples available for language {language} with the specified labels.\n")
            continue
        
        # Randomly select samples
        samples = reduced_df.sample(num_samples_to_show)
        
        print(f"Language: {language}\n")
        for _, row in samples.iterrows():
            print(f"Claim: {row['claim']}")
            print(f"Label: {row['label']}")
            print("Evidence:")
            for i in range(1, 6):
                evidence = row.get(f'evidence_{i}', None)
                if pd.notna(evidence) and evidence:
                    print(f"  Evidence {i}: {evidence}")
            print("\n" + "-"*50 + "\n")
            
# Display samples
display_samples_per_language(df)