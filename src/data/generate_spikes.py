from sklearn.decomposition import PCA
import numpy as np


def generate_random_spikes(data, n_participants):
    '''
    Returns the spikes for one participant
            Parameters:
                    data (array of shape (n_samples, n_features)): the samples of a participant
                    n_participants (int): the number of all the participants

            Returns:
                    spikes (array of shape (x, n_features)): the generated spikes for a participant
    '''
    variance = 0.7
    spikes_nb = max(data.shape[1] // n_participants, 1)
    pca = PCA(variance)
    dataset_pca = pca.fit_transform(data)
    generated_spikes = np.random.uniform(low=dataset_pca.min(axis=0),
                                         high=dataset_pca.max(axis=0),
                                         size=(spikes_nb, dataset_pca.shape[1]))
    return pca.inverse_transform(generated_spikes)


def generate_spikes(participants_data, global_data_features):
    '''
    Returns the generated spikes
            Parameters:
                    participants_data (array of shape (n_participants, n_samples, n_features)):
                        the samples of all the participants
                    global_data_features (int): the number of features for every participant

            Returns:
                    spikes (array of shape (x, n_features)): the generated spikes
    '''
    result = []
    for data in participants_data:
        result.append(generate_random_spikes(data, len(participants_data)))
    spikes = np.concatenate(result, axis=0)
    # reduce spikes for the sake of security
    if spikes.shape[0] >= global_data_features:
        spikes = spikes[:global_data_features - 1]
    return spikes
