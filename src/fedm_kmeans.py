from src.data.generate_dataset import generate_dataset
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from src.data.divide_dataset import divide_dataset
from src.data.generate_spikes import generate_spikes
from sklearn.metrics.cluster import adjusted_rand_score
from src.data.generate_spikes import get_number_of_participants_clusters
from src.data.k_analysis import get_number_of_clusters
from src.visualization.visualize import plot_kmeans_clusters

PARTICIPANTS = 2
TOTAL_GLOBAL_SAMPLES = 750
GLOBAL_CLUSTER_STD = 0.1
GLOBAL_CLUSTER_FEATURES = 2
DATASET_DIVISION_TYPE = 'iid'  # 'iid' 'non-iid points' 'non-iid clusters'
CENTERS = [[1,1], [-1,-1]]




# generate toy dataset
global_data, global_labels_true = generate_dataset(cluster_std=GLOBAL_CLUSTER_STD, n_features=GLOBAL_CLUSTER_FEATURES,
                                                   samples_nb=TOTAL_GLOBAL_SAMPLES,
                                                   centers=CENTERS,
                                                   type='blobs')
global_data_matrix = euclidean_distances(global_data)

plt.scatter(global_data[:, 0], global_data[:, 1])
plt.title('Global data')
plt.show()

# clustering with K-means
global_clusters_nb, _ = get_number_of_clusters(global_data)
global_clusterer = KMeans(n_clusters=global_clusters_nb, n_init='auto', random_state=10)
global_labels_kmeans = global_clusterer.fit_predict(global_data)
print(f'Global number of clusters:{global_clusters_nb}')
# plot_kmeans_clusters(global_data, global_labels_kmeans, global_clusterer.cluster_centers_, 'Global data')


# dividing global data by participants
participants_data, participants_labels = divide_dataset(global_data, PARTICIPANTS, global_labels_true,
                                                                    DATASET_DIVISION_TYPE, random_seed=42)

#### Participant Based Computation ####
spikes = generate_spikes(participants_data)
# print(f'Number of spikes:{len(spikes)}')
lsdm = [euclidean_distances(d, spikes) for d in participants_data]

#### Coordinator Based Computation ####
lsdm_concat = np.concatenate(lsdm)
fedm_matrix = euclidean_distances(lsdm_concat)
fedm_clusters = get_number_of_clusters(fedm_matrix)[0]
# print'fedm_clusters:')
fedm_model = KMeans(n_clusters=fedm_clusters, n_init='auto', random_state=10)
fedm_labels = fedm_model.fit_predict(fedm_matrix)

# correlations between distance matrix and FEDM
# cannot be used for 'non-iid clusters'
pearson_adm_fedm = np.corrcoef(global_data_matrix.flatten(), fedm_matrix.flatten())
spearman_adm_fedm = spearmanr(global_data_matrix.flatten(), fedm_matrix.flatten())
print('===============')
print(f'Pearson FEDM vs ADM: {pearson_adm_fedm[0, 1]:.3f}')
print(f'Spearman FEDM vs ADM: {spearman_adm_fedm.correlation:.3f}')
print('===============')
print(f'Rand_score KMeans vs true: {adjusted_rand_score(global_labels_true, global_labels_kmeans):.3f}')
print(f'Rand_score FEDM vs true: {adjusted_rand_score(participants_labels, fedm_labels):.3f}')

# labels are mixed for fedm in 'non-iid clusters'
if DATASET_DIVISION_TYPE == 'non-iid clusters':
    print(f'Rand_score FEDM vs KMeans: {adjusted_rand_score(fedm_labels, np.sort(global_labels_kmeans)[::-1]):.3f}')
else:
    print(f'Rand_score FEDM vs KMeans: {adjusted_rand_score(fedm_labels, global_labels_kmeans):.3f}')
#TODO dbscan not possible?

