import numpy as np
import random


def divide_dataset(global_dataset, participants_nb, labels, type, random_seed=42):
    '''
    Returns the generated blobs with labels
            Parameters:
                    global_dataset (int): the dataset
                    participants_nb (int): the total number of participants
                    labels (array): the corresponding labels of global_dataset
                    type (str): the type of dataset partitioning
                    random_seed (int): pass an int for reproducible output

            Returns:
                    (array of shape (n_participants, x, n_features)): the generated samples
                    new_labels (array of shape (labels,)): the new labels
    '''
    samples_nb = global_dataset.shape[0]
    samples_by_part = samples_nb // participants_nb
    # uniformly distributed samples
    if type == 'iid':
        divided_dataset = [global_dataset[(i - 1) * samples_by_part: i * samples_by_part]
                           for i in range(1, participants_nb)]
        divided_dataset.append(global_dataset[(participants_nb - 1) * samples_by_part:])
        new_labels = labels
    # less uniformly distributed samples
    elif type == 'non-iid points':
        random.seed(random_seed)
        start = 0
        end = 0
        divided_dataset = []
        for i in range(1, participants_nb):
            while end == start:
                end = random.randint(start, samples_by_part * i)
                # print('Warning! Infinite loop')
            divided_dataset.append(global_dataset[start: end])
            start = end
        divided_dataset.append(global_dataset[start:])
        new_labels = labels
    # samples assigned to participant by label
    # participants_nb is useless in this option
    elif type == 'non-iid clusters':
        clusters = set(labels)
        divided_dataset = [global_dataset[labels == c] for c in clusters]
        # new_labels = np.asarray([labels[labels == c] for c in clusters]).flatten()
        new_labels = np.sort(labels)
    else:
        raise ValueError('incorrect type')
    return np.asarray(divided_dataset, dtype=object), new_labels