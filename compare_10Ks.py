import os
import pandas as pd
import difflib

# Function to read the 10-K report from a text file
def read_10k_report(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to display differences between two texts
def display_differences(text1, text2):
    diff = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm='')
    for line in diff:
        print(line)

# Paths to the 10-K reports
report1_path = "data/2019/QTR2/20190401_10-K_edgar_data_8177_0001140361-19-006219.txt"
report2_path = "data/preprocessed.parquet"

# Read the first 10-K report
report1_content = read_10k_report(report1_path)

# Read the second 10-K report from the parquet file
df = pd.read_parquet(report2_path)
df['content'] = df.apply(lambda row: ' '.join([str(row[col]) if row[col] is not None else '' for col in df.columns if col.startswith('section_')]), axis=1)
report2_content = df.loc[df['filingUrl'].str.contains('0001140361-19-006219'), 'content'].values[0]

# Display differences between the two reports
display_differences(report1_content, report2_content)