from sklearn.decomposition import PCA
import numpy as np


def generate_random_spikes(data, n_participants):
    variance = 0.7
    spikes_nb = max(data.shape[1] // n_participants, 1)
    pca = PCA(variance)
    dataset_pca = pca.fit_transform(data)
    generated_spikes = np.random.uniform(low=dataset_pca.min(axis=0),
                                         high=dataset_pca.max(axis=0),
                                         size=(spikes_nb, dataset_pca.shape[1]))
    return pca.inverse_transform(generated_spikes)


def generate_spikes(participants_data, global_data_features):
    result = []
    for data in participants_data:
        result.append(generate_random_spikes(data, len(participants_data)))
    spikes = np.concatenate(result, axis=0)
    # reduce spikes for the sake of security
    if spikes.shape[0] >= global_data_features:
        spikes = spikes[:global_data_features - 1]
    return spikes
