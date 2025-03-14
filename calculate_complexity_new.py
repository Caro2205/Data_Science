import os
import json
import pandas as pd
import psutil


def print_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

root_folder = "data"
complexity_words_file = "complexity_words.txt"
output_file = "complexity_scores.json"

print_memory()

# Load complexity words
with open(complexity_words_file, "r") as file:
    complexity_words = set(word.strip().lower() for word in file.readlines())

complexity_scores = {}
# Load data
data_file = "data/final_data.parquet"

df = pd.read_parquet(data_file)
print(df.columns)

# combine single sections into one content column
df['content'] = df.apply(lambda row: ' '.join([str(row[col]) if row[col] is not None else '' for col in df.columns if col.startswith('section_')]), axis=1)
#iterate through all 10-K filings and calculate complexity score (percentage of complexity words)
for index, row in df.iterrows():
    print_memory()
    print(index)
    file_name = row['filingUrl']
    content = row['content'].lower()
    words = content.split()
    total_words = len(words)

    # Count occurrences of complexity words
    complexity_count = sum(1 for word in words if word in complexity_words)

    # Calculate complexity score
    if total_words > 0:
        complexity_score = (complexity_count / total_words) * 100
    else:
        complexity_score = 0

    complexity_scores[file_name] = complexity_score
    complexity_scores[file_name] = {
        "complexity_score": complexity_score,
        "cik": row['cik'],
        "filing_date": row['filedAt']
    }

    complexity_df = pd.DataFrame.from_dict(complexity_scores, orient='index')
    complexity_df.to_csv("complexity_scores_final.csv", index_label="file_name")

    
