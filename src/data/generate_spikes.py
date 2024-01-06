from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA


def get_number_of_clusters(participants_data, min_clusters=False):
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    silhouette_values = [[] for _ in range(participants_data.shape[0])]
    threshold = 0.6
    participant_nb = 0
    result = []
    for data in participants_data:
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
            cluster_labels = clusterer.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_values[participant_nb].append(silhouette_avg)
            # print(f'For participant_nb = {participant_nb} For n_clusters = {n_clusters} '
            #       f'The average silhouette_score={silhouette_avg}')
        participant_nb += 1
    silhouette_values = np.asarray(silhouette_values)
    # print(f'silhouette_values:{silhouette_values}')
    for participant_values in silhouette_values:
        # first occurrence of value greater than threshold
        # get minimum number of clusters
        if any(participant_values > threshold):
            # favors min number of clusters
            if min_clusters:
                result.append(np.argmax(participant_values > threshold) + range_n_clusters[0])
            # favors max number of clusters
            else:
                result.append(np.argmax(participant_values) + range_n_clusters[0])
        else:
            result.append(-1)
    return result


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
    n_clusters = get_number_of_clusters(participants_data)
    print(f'determined clusters for every participant:{n_clusters}')
    result = []
    for n, data in zip(n_clusters, participants_data):
        if n == -1:
            result.append(generate_random_spikes(data))
        else:
            kmeans = KMeans(n_clusters=n, n_init='auto', random_state=10).fit(data)
            result.append(kmeans.cluster_centers_)
    return np.concatenate(result, axis=0)
