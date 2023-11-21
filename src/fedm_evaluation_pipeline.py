from data.generate_dataset import generate_dataset
from visualization.visualize import plot_db_clusters
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from data.divide_dataset import generate_participants_data

PARTICIPANTS = 2
TOTAL_GLOBAL_SAMPLES = 750
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 10
CENTERS = [[1, 1], [-1, -1]]
SPIKES = [[1, 1]]
DATASET_DIVISION_TYPE = 'non-iid clusters'  # 'iid' 'non-iid points' 'non-iid clusters'


def evaluation(labels_true, labels_predicted):
    return f1_score(labels_true, labels_predicted, average='micro')

# generate toy dataset
global_data, global_labels_true = generate_dataset(samples_nb=TOTAL_GLOBAL_SAMPLES, type='blobs', n_features=4,
                                                   centers=CENTERS, cluster_std=0.3)
global_data_dm = euclidean_distances(global_data)
plt.scatter(global_data[:, 0], global_data[:, 1])
plt.title('global data')
plt.show()

# clustering with DBSCAN
db = DBSCAN(metric='precomputed', eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(global_data_dm)
plot_db_clusters(db, global_data)

# clustering with Federated Learning
participants_data = generate_participants_data(global_data, PARTICIPANTS, db.labels_, DATASET_DIVISION_TYPE, random_seed=42)

# plotting only 1 and 2 dimension
for d in participants_data:
    plt.scatter(d[:, 0], d[:, 1])
plt.title('participants data')
plt.show()
# plt.savefig('../reports/figures/data_division/' + DATASET_DIVISION_TYPE)

#### Participant Based Computation ####
# TODO: spikes generation
spikes = SPIKES
lsdm = [euclidean_distances(d, spikes) for d in participants_data]

#### Coordinator Based Computation ####
lsdm_concat = np.concatenate(lsdm)
fedm = euclidean_distances(lsdm_concat)

fedm_clustering = DBSCAN(metric='precomputed', eps=0.1, min_samples=3).fit(fedm)
plot_db_clusters(fedm_clustering, np.concatenate(participants_data))

# correlations between distance matrix and FEDM
pearson_adm_fedm = np.corrcoef(global_data_dm.flatten(), fedm.flatten())
spearman_adm_fedm = spearmanr(global_data_dm.flatten(), fedm.flatten())
print(f'Pearson FEDM vs ADM: {pearson_adm_fedm[0, 1]:.3f}')
print(f'Spearman FEDM vs ADM: {spearman_adm_fedm.correlation:.3f}')

# TODO: Hungarian algorithm – change of labels
# TODO: right labels
print(f'F1_score FEDM vs true: {evaluation(global_labels_true, fedm_clustering.labels_)}')
print(f'F1_score DBSCAN vs true: {evaluation(global_labels_true, db.labels_)}')