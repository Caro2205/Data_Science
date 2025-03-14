import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('/home/caro/DS_complexity/complexity_scores_new.csv')

# Function to reformat the date
def reformat_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
    return date_obj.strftime('%Y%m%d')

# Apply the function to the filing_date column
df['filing_date'] = df['filing_date'].apply(reformat_date)

# Save the updated DataFrame to a new CSV file
df.to_csv('/home/caro/DS_complexity/complexity_scores_new_reformatted.csv', index=False)