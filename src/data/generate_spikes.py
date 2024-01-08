from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from src.data.k_analysis import get_number_of_participants_clusters


def generate_random_spikes(data):
    variance = 0.7
    spikes_nb = 1
    pca = PCA(variance)
    dataset_pca = pca.fit_transform(data)
    generated_spikes = np.random.uniform(low=dataset_pca.min(axis=0),
                                         high=dataset_pca.min(axis=0),
                                         size=(spikes_nb, dataset_pca.shape[1]))
    return pca.inverse_transform(generated_spikes)


def generate_spikes(participants_data):
    n_clusters = get_number_of_participants_clusters(participants_data)
    print(f'determined clusters for every participant:{n_clusters}')
    result = []
    for n, data in zip(n_clusters, participants_data):
        if n == -1:
            result.append(generate_random_spikes(data))
        else:
            kmeans = KMeans(n_clusters=n, n_init='auto', random_state=10).fit(data)
            result.append(kmeans.cluster_centers_)
    return np.concatenate(result, axis=0)
