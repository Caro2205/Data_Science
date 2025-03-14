import pandas as pd
import json
import re
# Load new CSV data
calculated_data = pd.read_csv("complexity_scores_new_reformatted.csv")

# Extract relevant columns
calculated_data = calculated_data[["complexity_score", "cik", "filing_date"]]

# Rename columns to match the existing code
calculated_data.rename(columns={"filing_date": "filingdate"}, inplace=True)

# Load CSV data
original_data = pd.read_csv("Loughran_McDonald_Complexity.csv")


# Convert JSON data to DataFrame
json_df = pd.DataFrame(calculated_data)
# Extract relevant columns
json_df = json_df[["complexity_score", "cik", "filingdate"]]
# Rename columns to avoid conflicts
json_df.rename(columns={"complexity_score": "json_complexity_score"}, inplace=True)

# Debug: Check columns and types
print("CSV columns:", original_data.columns)
print("JSON DataFrame columns:", json_df.columns)

# Ensure 'cik' column is of the same type
original_data["cik"] = original_data["cik"].astype(int)
json_df["cik"] = json_df["cik"].astype(int)

original_data["filingdate"] = original_data["filingdate"].astype(int)
json_df["filingdate"] = json_df["filingdate"].astype(int)

# Debug: Check unique CIKs
#print("Unique CIKs in CSV:", original_data["cik"].unique())
#print("Unique CIKs in calculated Data:", json_df["cik"].unique())

# Merge DataFrames
merged_data = pd.merge(original_data, json_df, on=["cik", "filingdate"], how="inner")

# Save the merged data for inspection
merged_data.to_csv("merged_complexity_scores_final.csv", index=False)

# Calculate and print the median of both complexity measurements
original_median = merged_data['complexity'].median()
calculated_median = merged_data['json_complexity_score'].median()

print(f"Median of original complexity scores: {original_median}")
print(f"Median of newly calculated complexity scores: {calculated_median}")

# Debug: Check merged data
print("Merged data preview:")
print(merged_data.head())

import matplotlib.pyplot as plt

# Plotting complexity scores
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['complexity'], merged_data['json_complexity_score'], color='blue', alpha=0.6)

# Labels and title
#plt.title("Complexity Scores Comparison", fontsize=14)
plt.xlabel("Original Complexity Scores by Loughran and McDonald", fontsize=12)
plt.ylabel("Our Calculated Complexity", fontsize=12)

# Show grid
plt.grid(True)

# Display plot
#plt.show()
plt.savefig('complexity_scores_comparison.png')

print(f"correlation of original and newly calculated complexity scores: {merged_data['complexity'].corr(merged_data['json_complexity_score'])}")