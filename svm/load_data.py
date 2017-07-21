__author__ = 'PC-LiNing'

import numpy as np


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    len(data) is not divisible by batch_size .
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data)/batch_size)
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # shuffle_indices = train_shuffle[epoch]
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def extract_data(csv_file):
    out = np.loadtxt(csv_file, delimiter=',')
    labels = out[:, 0]
    labels = labels.reshape(labels.size, 1)
    # convert labels to +1,-1
    labels[labels == 0] = -1
    data = out[:, 1:]
    return data, labels

"""
data,labels = extract_data('linearly_separable_data.csv')
print(data.shape)
"""
