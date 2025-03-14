import pandas as pd

# Load the parquet file
df = pd.read_parquet('data/final_data.parquet')

# Print the length of the dataframe
print(len(df))