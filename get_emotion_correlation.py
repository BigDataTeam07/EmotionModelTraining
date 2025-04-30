import pandas as pd
import numpy as np

train_file = "data\\train.tsv"
emotions_file = "data\\emotions.txt"
output_file = "data\\emotion_correlation.tsv"

with open(emotions_file, "r", encoding="utf-8") as f:
    emotions = [line.strip() for line in f]
num_labels = len(emotions)

# co-occurrence matrix
co_matrix = np.zeros((num_labels, num_labels), dtype=int)

# count co-occurrences
df = pd.read_csv(train_file, sep="\t", header=None, names=["text", "labels", "ids"])
df.drop(columns=["ids"], inplace=True)  # delete ids column

for label_str in df["labels"]:
    try:
        indices = list(map(int, str(label_str).split(",")))
        for i in indices:
            for j in indices:
                co_matrix[i][j] += 1
    except Exception as e:
        print(f"skip unexpected line: {label_str} - {e}")

# normalize the co-occurrence matrix and set diagonal to 0
max_val = co_matrix.max()
norm_matrix = co_matrix / max_val if max_val > 0 else co_matrix
np.fill_diagonal(norm_matrix, 0.0)

df_corr = pd.DataFrame(norm_matrix, index=emotions, columns=emotions)
df_corr.to_csv(output_file, sep="\t", float_format="%.4f")

print(f"saved matrix to: {output_file}")
