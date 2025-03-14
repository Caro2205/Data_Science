input_file = 'financial_phrases_all.txt'
output_file = 'financial_phrases_all_updated.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        parts = line.strip().split('\t')
        if parts:
            sentiment_label = int(parts[-1])
            new_sentiment_label = sentiment_label + 1
            parts[-1] = str(new_sentiment_label)
            outfile.write('\t'.join(parts) + '\n')