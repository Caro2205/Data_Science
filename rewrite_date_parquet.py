import pandas as pd
from datetime import datetime
# Read the Parquet file
df = pd.read_parquet('/home/caro/DS_complexity/data/final_data.parquet')

# Function to reformat the date
def reformat_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.strftime('%Y%m%d')

# Apply the function to the Filing Date column
df['Filing Date'] = df['Filing Date'].apply(reformat_date)

# Save the updated DataFrame to a new Parquet file
df.to_parquet('/home/caro/DS_complexity/data/final_data_reformatted.parquet', index=False)
