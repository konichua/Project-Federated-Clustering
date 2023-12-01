from data.generate_dataset import generate_dataset
from visualization.visualize import plot_db_clusters
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from data.divide_dataset import generate_participants_data
import participant_utils as pu
import coordinator_utils as cu

PARTICIPANTS = 2
TOTAL_GLOBAL_SAMPLES = 750
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 10
CENTERS = [[1, 1], [-1, -1]]


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
D1, D2 = generate_participants_data(global_data, PARTICIPANTS, db.labels_, 'non_uniform', random_seed=42)
# D1 = part_data[0]
# D2 = part_data[1]
plt.scatter(D1[:,0], D1[:,1])
plt.scatter(D2[:,0], D2[:,1])
# plt.scatter(D3[:,0], D3[:,1])
plt.title('blue data is from D1, orange from D2')
plt.show()


spikes = [CENTERS[0]]
lsdm_d1 = euclidean_distances(D1, spikes)
lsdm_d2 = euclidean_distances(D2, spikes)

#### Coordinator Based Computation ####
LSDM_concat = np.concatenate([lsdm_d1, lsdm_d2])
fedm = euclidean_distances(LSDM_concat)

# Regression analysis for each participant
slope_intercept_D1 = pu.regression_per_client(data=D1, euc_dist_data_spike=lsdm_d1, regressor="Linear")
slope_intercept_D2 = pu.regression_per_client(data=D2, euc_dist_data_spike=lsdm_d2, regressor="Huber")
# Compute global Euclidean distances
global_true_euc_dist = euclidean_distances(global_data)
global_fed_euc_dist = cu.calc_fed_euc_dist([lsdm_d1, lsdm_d2,])

# Construct global matrices for regression parameters
MxCx = []
MxCx.append(slope_intercept_D1)
MxCx.append(slope_intercept_D2)

# Calculate predicted global Euclidean distances
global_Mx, global_Cx = cu.construct_global_Mx_Cx_matrix(MxCx, [lsdm_d1.shape[0], lsdm_d2.shape[0]])
global_pred_euc_dist = cu.calc_pred_dist_matrix(global_Mx, global_fed_euc_dist, global_Cx)

# correlations between distance matrix and FEDM
pearson_adm_pedm = np.corrcoef(global_data_dm.flatten(), global_pred_euc_dist.flatten())
spearman_adm_pedm = spearmanr(global_data_dm.flatten(), global_pred_euc_dist.flatten())
print(f'Pearson PEDM vs ADM: {pearson_adm_pedm[0, 1]}')
print(f'Spearman PEDM vs ADM: {spearman_adm_pedm.correlation}')


#print(f'F1_score PEDM vs true: {evaluation(global_labels_true, pedm_clustering.labels_)}')
#print(f'F1_score DBSCAN vs true: {evaluation(global_labels_true, db.labels_)}')