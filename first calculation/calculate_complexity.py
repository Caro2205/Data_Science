import os
import json

root_folder = "data"
complexity_words_file = "complexity_words.txt"
output_file = "complexity_scores.json"

# Load complexity words
with open(complexity_words_file, "r") as file:
    complexity_words = set(word.strip().lower() for word in file.readlines())

complexity_scores = {}

for year_folder in os.listdir(root_folder):
    year_path = os.path.join(root_folder, year_folder)
    if os.path.isdir(year_path):  # Check if it's a directory
        for quarter_folder in os.listdir(year_path):
            quarter_path = os.path.join(year_path, quarter_folder)
            print(f'Processing {quarter_path}')
            if os.path.isdir(quarter_path):  # Check if it's a directory
                for file_name in os.listdir(quarter_path):
                    file_path = os.path.join(quarter_path, file_name)
                    if os.path.isfile(file_path) and file_name.endswith(".txt"):  # Check if it's a text file
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().lower()
                            words = content.split()
                            total_words = len(words)

                            # Count occurrences of complexity words
                            complexity_count = sum(1 for word in words if word in complexity_words)

                            # Calculate complexity score
                            if total_words > 0:
                                complexity_score = (complexity_count / total_words) * 100
                            else:
                                complexity_score = 0

                            # Store the result
                            complexity_scores[file_name] = complexity_score

# Save results to a JSON file
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(complexity_scores, json_file, indent=4)

print(f"Complexity scores have been saved to {output_file}")