import numpy as np
from sklearn.model_selection import train_test_split


def split_data(task):
    with np.load(f'./data/{task}-positive.npz') as pos,\
         np.load(f'./data/{task}-negative.npz') as neg:
        pos_data = pos['arr_0']
        neg_data = neg['arr_0']

    input_ids = np.vstack((
        pos_data,
        neg_data
    ))

    labels = np.hstack((
        np.ones(len(pos_data)),
        np.zeros(len(neg_data))
    ))

    train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
        input_ids, labels, test_size=0.1
    )

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_val_inputs, train_val_labels, test_size=0.1
    )

    np.savez(
        f'./data/{task}/train.npz',
        inputs=train_inputs,
        labels=train_labels
    )

    np.savez(
        f'./data/{task}/val.npz',
        inputs=val_inputs,
        labels=val_labels
    )

    np.savez(
        f'./data/{task}/test.npz',
        inputs=test_inputs,
        labels=test_labels
    )


def combine(inputs, labels, new_inputs, new_labels):
    new_inputs = np.vstack(new_inputs)
    new_labels = np.hstack(new_labels)

    inputs = np.vstack((inputs, new_inputs))
    labels = np.hstack((labels, new_labels))

    return inputs, labels


def random_replace(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        to_replace = int(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, to_replace, replace=False)
        ip[indices] = np.random.choice(np.arange(5, 25), to_replace, replace=True)

        new_inputs.append(ip)
        new_labels.append(label)

    return combine(inputs, labels, new_inputs, new_labels)


def augment_data(task):
    with np.load(f'./data/{task}/train.npz') as train:
        inputs = train['inputs']
        labels = train['labels']

    inputs, labels = random_replace(inputs, labels, 0.05)

    np.savez(
        f'./data/{task}/train.npz',
        inputs=inputs,
        labels=labels
    )


def main():
    # split_data('hemo')
    split_data('sol')
    # split_data('nf')

    augment_data('sol')

main()
