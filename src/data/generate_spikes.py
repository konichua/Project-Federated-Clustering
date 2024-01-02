from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def get_number_of_clusters(participants_data):
    range_n_clusters = [2, 3, 4]  # [2, 3, 4, 5, 6, 7, 8, 9, 10]
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
            print(f'For participant_nb = {participant_nb} For n_clusters = {n_clusters} '
                  f'The average silhouette_score={silhouette_avg}')
        participant_nb += 1
    silhouette_values = np.asarray(silhouette_values)
    # print(f'silhouette_values:{silhouette_values}')
    for participant_values in silhouette_values:
        # first occurrence of value greater than threshold
        # get minimum number of clusters
        if any(participant_values > threshold):
            result.append(np.argmax(participant_values > threshold) + range_n_clusters[0])
        else:
            result.append(-1)
    return result


def generate_spikes(participants_data):
    n_clusters = get_number_of_clusters(participants_data)
    print(f'determined clusters for every participant:{n_clusters}')
    result = []

    for n, data in zip(n_clusters, participants_data):
        if n == -1:
            continue
        kmeans = KMeans(n_clusters=n, n_init='auto', random_state=10).fit(data)
        result.append(kmeans.cluster_centers_)
    return np.concatenate(np.asarray(result))
