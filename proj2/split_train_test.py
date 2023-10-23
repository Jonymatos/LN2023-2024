import random

# File paths
input_file = 'proj2/train.txt'
train_output_file = 'proj2/train_split.txt'
test_output_file = 'proj2/test.txt'

# Load all lines from the file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Shuffle lines
random.shuffle(lines)

# Split lines into train and test
split_index = int(0.8 * len(lines))
train_lines = lines[:split_index]
test_lines = lines[split_index:]

# Save train split
with open(train_output_file, 'w') as f:
    f.writelines(train_lines)

# Save test split
with open(test_output_file, 'w') as f:
    f.writelines(test_lines)

print(f"Train data saved to {train_output_file}")
print(f"Test data saved to {test_output_file}")