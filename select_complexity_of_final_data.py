import pandas as pd

# Load the complexity scores data
complexity_scores = pd.read_csv('/home/caro/DS_complexity/complexity_scores_new_reformatted.csv')

# Load the final data
final_data = pd.read_parquet('/home/caro/DS_complexity/data/final_data_reformatted.parquet')
print(final_data.head())

# Convert 'cik' columns to object type
#complexity_scores['cik'] = complexity_scores['cik'].astype(str)
final_data['CIK'] = final_data['CIK'].astype('Int64')
complexity_scores['filing_date'] = complexity_scores['filing_date'].astype(str)
final_data['Filing Date'] = final_data['Filing Date'].astype(str)

# Merge the dataframes on 'cik' and 'filing_date'
merged_data = pd.merge(complexity_scores, final_data, how='inner', left_on=['cik', 'filing_date'], right_on=['CIK', 'Filing Date'])

# Select the relevant columns
filtered_data = merged_data[['file_name', 'complexity_score', 'cik', 'filing_date']]

# Save the filtered data to a new CSV file
filtered_data.to_csv('/home/caro/DS_complexity/complexity_scores_final.csv', index=False)