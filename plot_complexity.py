import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# Load JSON data
with open("complexity_scores.json", "r") as file:
    data = json.load(file)

# Extract year and scores
year_scores = defaultdict(list)
for filename, score in data.items():
    match = re.match(r"(\d{4})", filename)  # Extract year using regex
    if match:
        year = int(match.group(1))
        year_scores[year].append(score * 100)  # Convert to percentage

# Prepare data for plotting
years = sorted(year_scores.keys())
scores = [score for year in years for score in year_scores[year]]

# Slightly jitter the years for better scatter visualization
year_labels = [year + random.uniform(-0.2, 0.2) for year in years for _ in year_scores[year]]

# Calculate mean and median for each year
means = [np.mean(year_scores[year]) for year in years]
medians = [np.median(year_scores[year]) for year in years]
print(medians)

# Plot scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(year_labels, scores, alpha=0.5, label="Individual Scores", color="blue")
plt.plot(years, means, marker="o", color="green", label="Mean (by Year)", linestyle="-", linewidth=2)
plt.plot(years, medians, marker="s", color="red", label="Median (by Year)", linestyle="--", linewidth=2)

# Add labels and legend
plt.title("Complexity Scores by Year")
plt.xlabel("Year")
plt.ylabel("Complexity Score (%)")
plt.xticks(years)
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
