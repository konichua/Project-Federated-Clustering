from src.data.generate_dataset import generate_dataset
from src.visualization.visualize import plot_db_clusters
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from src.data.divide_dataset import divide_dataset
from src.data.generate_spikes import generate_spikes
from sklearn.metrics.cluster import adjusted_rand_score
from src.data.generate_spikes import get_number_of_clusters
from sklearn.metrics import silhouette_score

PARTICIPANTS = 5
TOTAL_GLOBAL_SAMPLES = 750
GLOBAL_CLUSTER_STD = 0.9
GLOBAL_CLUSTER_FEATURES = 2
DATASET_DIVISION_TYPE = 'iid'  # 'iid' 'non-iid points' 'non-iid clusters'


def get_clusters_number(data, min_clusters=False):
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    silhouette_values = []
    threshold = 0.6
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_values.append(silhouette_avg)
    silhouette_values = np.asarray(silhouette_values)
    # print(f'silhouette_values:{silhouette_values}')
    if any(silhouette_values > threshold):
        # favors min number of clusters
        if min_clusters:
            return np.argmax(silhouette_values > threshold) + range_n_clusters[0]
        # favors max number of clusters
        return np.argmax(silhouette_values) + range_n_clusters[0]
    return 1

# generate toy dataset
global_data, global_labels_true = generate_dataset(cluster_std=GLOBAL_CLUSTER_STD, n_features=GLOBAL_CLUSTER_FEATURES,
                                                   samples_nb=TOTAL_GLOBAL_SAMPLES,
                                                   type='blobs')
global_data_matrix = euclidean_distances(global_data)

plt.scatter(global_data[:, 0], global_data[:, 1])
plt.title('global data')
plt.show()

# clustering with K-means
global_clusterer = KMeans(n_clusters=get_clusters_number(global_data), n_init='auto', random_state=10)
global_labels_kmeans = global_clusterer.fit_predict(global_data)
print(f'get_clusters_number(global_data):{get_clusters_number(global_data)}')


# dividing global data by participants
participants_data, participants_labels = divide_dataset(global_data, PARTICIPANTS, global_labels_true,
                                                                    DATASET_DIVISION_TYPE, random_seed=42)

#### Participant Based Computation ####
spikes = generate_spikes(participants_data)
print(f'spikes{spikes}')
lsdm = [euclidean_distances(d, spikes) for d in participants_data]

#### Coordinator Based Computation ####
lsdm_concat = np.concatenate(lsdm)
fedm_matrix = euclidean_distances(lsdm_concat)
fedm_clustering = KMeans(n_clusters=get_clusters_number(fedm_matrix), n_init='auto', random_state=10)
fedm_labels = fedm_clustering.fit_predict(fedm_matrix)

# print(f'global_labels_true:{global_labels_true}')
# print(f'global_labels_kmeans:{global_labels_kmeans}')
# correlations between distance matrix and FEDM
pearson_adm_fedm = np.corrcoef(global_data_matrix.flatten(), fedm_matrix.flatten())
spearman_adm_fedm = spearmanr(global_data_matrix.flatten(), fedm_matrix.flatten())
print('===============')
print(f'Pearson FEDM vs ADM: {pearson_adm_fedm[0, 1]:.3f}')
print(f'Spearman FEDM vs ADM: {spearman_adm_fedm.correlation:.3f}')
print('===============')
print(f'Rand_score KMeans vs true: {adjusted_rand_score(global_labels_true, global_labels_kmeans):.3f}')
print(f'Rand_score FEDM vs true: {adjusted_rand_score(participants_labels, fedm_labels):.3f}')
#TODO fedm - mixed, while kmeans - not mixed  (it works for iid case anyway)
#TODO silhouette analysis
print(f'Rand_score FEDM vs KMeans: {adjusted_rand_score(fedm_labels, global_labels_kmeans):.3f}')
