import numpy as np
import os
from sklearn.model_selection import train_test_split

task_name = 'tox'

# read data
seqs, labels = [], []
with open('/Users/irene/PeptideBERT/3864.csv', 'r') as f:
    for line in f.readlines()[1:]:
        seq, label = line.strip().split(',')
        seqs.append(seq)
        labels.append(int(label))
# Assuming 'seqs' is a list of sequences and 'labels' is a list of labels
MAX_LEN = max(map(len, seqs))

# Convert to tokens and pad the sequences to MAX_LEN
mapping = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
     'A','G','V','E','S','I','K','R','D','T','P','N',
     'Q','F','Y','M','H','C','W'],
    range(30)
))

pos_data, neg_data = [], []
for i in range(len(seqs)):
    seq = [mapping[c] for c in seqs[i]]  # Convert each sequence to tokens
    seq.extend([0] * (MAX_LEN - len(seq)))  # Pad to MAX_LEN
    if labels[i] == 1:
        pos_data.append(seq)
    else:
        neg_data.append(seq)

# Convert to numpy arrays with shape
pos_data = np.array(pos_data)
neg_data = np.array(neg_data)

# Check the shape of pos_data and neg_data
print(f"Positive data shape: {pos_data.shape}")
print(f"Negative data shape: {neg_data.shape}")

# Now stack the arrays correctly
input_ids = np.vstack((pos_data, neg_data))
labels = np.hstack((np.ones(len(pos_data)), np.zeros(len(neg_data))))

# Check the final shape of input_ids and labels
print(f"Input_ids shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")


task_name = 'tox'

def split_data(task):
    # Load positive and negative data
    with np.load(f'/Users/irene/PeptideBERT/data/{task_name}-positive-3864.npz') as pos, \
         np.load(f'/Users/irene/PeptideBERT/data/{task_name}-negative-3864.npz') as neg:
        pos_data = pos['arr_0']
        neg_data = neg['arr_0']

    # Check the shapes of pos_data and neg_data
    print(f"Positive data shape: {pos_data.shape}")
    print(f"Negative data shape: {neg_data.shape}")

    # Combine positive and negative data
    input_ids = np.vstack((pos_data, neg_data))
    labels = np.hstack((np.ones(len(pos_data)), np.zeros(len(neg_data))))

    # Check the shape of input_ids and labels
    print(f"Input_ids shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Split into train/validation/test sets
    train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
        input_ids, labels, test_size=0.1
    )

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_val_inputs, train_val_labels, test_size=0.1
    )

    # Create directory if it doesn't exist
    if not os.path.exists(f'/Users/irene/PeptideBERT/data/{task}'):
        os.mkdir(f'/Users/irene/PeptideBERT/data/{task}')

    # Save the data splits
    np.savez(
        f'/Users/irene/PeptideBERT/data/{task}/train.npz',
        inputs=train_inputs,
        labels=train_labels
    )

    np.savez(
        f'/Users/irene/PeptideBERT/data/{task}/val.npz',
        inputs=val_inputs,
        labels=val_labels
    )

    np.savez(
        f'/Users/irene/PeptideBERT/data/{task}/test.npz',
        inputs=test_inputs,
        labels=test_labels
    )

# Call the split_data function for debugging
split_data('tox')
