from data.generate_dataset import generate_dataset
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from data.divide_dataset import generate_participants_data
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


PARTICIPANTS = 2
TOTAL_GLOBAL_SAMPLES = 750
KMEANS_N_CLUSTERS = 2  # Set the desired number of clusters for KMeans
CENTERS = [[1, 1], [-1, -1]]


def evaluation(labels_true, labels_predicted):
    return f1_score(labels_true, labels_predicted, average='micro')

# generate toy dataset
global_data, global_labels_true = generate_dataset(samples_nb=TOTAL_GLOBAL_SAMPLES, type='blobs', n_features=4,
                                                   centers=CENTERS, cluster_std=0.3)



# ... (your existing code)

# clustering with Federated Learning


# ... (the rest of your existing code)


# ... (your existing code)

# clustering with KMeans for Global Clustering
kmeans_global = KMeans(n_clusters=len(CENTERS), random_state=42).fit(global_data)
centers_global = kmeans_global.cluster_centers_
labels_global = kmeans_global.labels_
D1, D2 = generate_participants_data(global_data, PARTICIPANTS, labels_global, 'non_uniform', random_seed=42)

# Scatter plot for global clustering
plt.scatter(global_data[:, 0], global_data[:, 1], c=labels_global, cmap='viridis')
plt.scatter(centers_global[:, 0], centers_global[:, 1], marker='X', s=200, color='red', label='Centroids')
plt.title('Global KMeans Clustering')
plt.legend()
plt.show()

# clustering with KMeans for Federated Learning
centroids_D1 = kmeans_global.cluster_centers_  # Use centroids from global clustering as spikes for D1
centroids_D2 = kmeans_global.cluster_centers_  # Use centroids from global clustering as spikes for D2

# Generate spikes for each participant
spikes_D1, _ = pairwise_distances_argmin_min(D1, centroids_D1)
spikes_D2, _ = pairwise_distances_argmin_min(D2, centroids_D2)

# Compute LSDM for each participant
lsdm_d1 = euclidean_distances(D1, spikes_D1.reshape(1, -1))
lsdm_d2 = euclidean_distances(D2, spikes_D2.reshape(1, -1))

# Concatenate LSDMs
LSDM_concat = np.concatenate([lsdm_d1, lsdm_d2])

# Compute FEDM
fedm = euclidean_distances(LSDM_concat)

# ... (the rest of your existing code)

# KMeans for Federated Clustering
kmeans_fedm = KMeans(n_clusters=len(CENTERS), random_state=42).fit(fedm)
# Scatter plot for federated clustering
plt.scatter(global_data[:, 0], global_data[:, 1], c=kmeans_fedm.labels_, cmap='viridis')
plt.scatter(kmeans_fedm.cluster_centers_[:, 0], kmeans_fedm.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids')
plt.title('Federated KMeans Clustering')
plt.legend()
plt.show()

# Evaluate F1 Score for Federated KMeans vs True Labels
print(f'F1_score Federated KMeans vs true: {evaluation(global_labels_true, kmeans_fedm.labels_)}')