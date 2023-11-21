import numpy as np
import random


def generate_participants_data(global_dataset, participants_nb, labels, type, random_seed=42):
    samples_nb = global_dataset.shape[0]

    if participants_nb < 2:
        raise ValueError('incorrect number of participants')
    # uniformly distributed samples
    if type == 'iid':
        samples_by_part = samples_nb // participants_nb
        divided_dataset = [global_dataset[(i - 1) * samples_by_part: i * samples_by_part]
                           for i in range(1, participants_nb)]
        divided_dataset.append(global_dataset[(participants_nb - 1) * samples_by_part:])
    # less uniformly distributed samples
    elif type == 'non-iid points':
        random.seed(random_seed)
        start = 0
        divided_dataset = list(np.zeros(participants_nb))
        for i in range(participants_nb - 1):
            end = random.randint(start, samples_nb)
            divided_dataset[i] = global_dataset[start: end]
            start = end
        divided_dataset[participants_nb - 1] = global_dataset[start:]
    # samples assigned to participant by label
    # noise data goes to the first participant
    # participants_nb is useless in this option
    elif type == 'non-iid clusters':
        clusters = set(labels)
        divided_dataset = [global_dataset[labels == c] for c in clusters if c != -1]
        divided_dataset[0] = np.concatenate([divided_dataset[0], global_dataset[labels == -1]])
    else:
        raise ValueError('incorrect type')
    return np.asarray(divided_dataset, dtype=object)