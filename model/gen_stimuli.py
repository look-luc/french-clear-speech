import pandas as pd

# Load the dataset
df = pd.read_csv('metadata_deduplicated.csv')

# Remove duplicate rows based on the 'text' column
deduped_df = df.drop_duplicates(subset=['text'])

# Save the result to a new CSV file
deduped_df.to_csv('metadata_deduplicated_cleaned.csv', index=False)

print(f"Original shape: {df.shape}")
print(f"Deduplicated shape: {deduped_df.shape}")
