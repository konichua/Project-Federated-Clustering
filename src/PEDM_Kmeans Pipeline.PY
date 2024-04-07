from data.generate_dataset import generate_dataset
from visualization.visualize import plot_db_clusters
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from data.divide_dataset import divide_dataset
import participant_utils as pu
import coordinator_utils as cu
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

PARTICIPANTS = 2
TOTAL_GLOBAL_SAMPLES = 1000
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 10
CENTERS = [[1, 1], [-1, -1]]
DATASET_DIVISION_TYPE='iid'

def evaluation(labels_true, labels_predicted):
    return f1_score(labels_true, labels_predicted, average='micro')

def evaluation(labels_true, labels_predicted):
    return f1_score(labels_true, labels_predicted, average='micro')

# generate toy dataset
global_data, global_labels_true = generate_dataset(samples_nb=TOTAL_GLOBAL_SAMPLES, type='blobs', n_features=10,
                                                   centers=CENTERS, cluster_std=0.3)
unique_labels = np.unique(global_labels_true)

global_data_dm = euclidean_distances(global_data)
plt.scatter(global_data[:, 0], global_data[:, 1])
plt.title('global data')
plt.show()


# clustering with DBSCAN
db = DBSCAN(metric='precomputed', eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(global_data_dm)
plot_db_clusters(db, global_data)

# clustering with Federated Learning
D1, D2 = divide_dataset(global_data, PARTICIPANTS, db.labels_, DATASET_DIVISION_TYPE, random_seed=42)
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


# correlations between distance matrix and FEDM
pearson_adm_fedm = np.corrcoef(global_data_dm.flatten(), fedm.flatten())
spearman_adm_fedm = spearmanr(global_data_dm.flatten(), fedm.flatten())
#print(f'Pearson FEDM vs ADM: {pearson_adm_fedm[0, 1]}')
#print(f'Spearman FEDM vs ADM: {spearman_adm_fedm.correlation}')

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


pedm_clustering = DBSCAN(metric='precomputed', eps=0.3, min_samples=10).fit(global_pred_euc_dist)

lsdm_concat = np.concatenate(global_pred_euc_dist)
#pedm_matrix = euclidean_distances(lsdm_concat)
#pedm_clusters = gold_centers
pedm_model = KMeans(n_clusters=len(unique_labels),n_init=10, init='k-means++', random_state=0)
pedm_labels = pedm_model.fit_predict(global_fed_euc_dist)
num_clusters = len(np.unique(pedm_labels))

''''''
#Keams from here

label = KMeans(n_clusters=len(unique_labels), random_state=0).fit_predict(global_true_euc_dist)
#Getting unique labels
#global_labels_kmeans = label.fit_predict(global_data)
u_labels_2 = np.unique(label)
pred_label_gtdm =  np.array(label).tolist()


''''''
clustered_dataset_2d = cu.perform_MDS(2, global_data)

''''''''
label = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(global_pred_euc_dist)
#Getting unique labels
u_labels_2 = np.unique(label)
pred_label_2 =  np.array(label).tolist()

plt.figure(figsize=(15,15))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.subplot(325)
plt.title("Clustering with predicted distance matrix", fontsize='medium')
for i in u_labels_2:
    plt.scatter(clustered_dataset_2d[label == i , 0] , clustered_dataset_2d[label == i , 1] , label = i)
#plt.scatter(spikes[:,0] , spikes[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

cu.unsupervised_evaluation_scores(global_pred_euc_dist, "Global Predicted Distance Matrix",  pred_label_gtdm, pred_label_2, adj_rand=True, adj_mutual_info=True, f1=True, silhouette=True, davies_bouldin=False)

#cu.plotDistanceMatrix(global_fed_euc_dist, title="Federated Global Distance Matrix")
#cu.plotDistanceMatrix(global_true_euc_dist, title="True Global Distance Matrix")
#cu.plotDistanceMatrix(global_pred_euc_dist, title="Predicted Global Distance Matrix")

cu.pearson_corr_coeff(global_true_euc_dist, global_fed_euc_dist, global_pred_euc_dist)


# Printing results for KMeans clustering
print("Results for KMeans Clustering:")
print(f"Adjusted Rand Score (KMeans vs True): {adjusted_rand_score(global_labels_true, pred_label_gtdm):.3f}")
print(f"Adjusted Rand Score (KMeans vs PEDM Labels): {adjusted_rand_score(pedm_labels, pred_label_2):.3f}")

#End Kmeans here


# Calculate Pearson correlation between PEDM and ADM
pearson_adm_pedm = np.corrcoef(global_pred_euc_dist.flatten(), global_data_dm.flatten())[0, 1]

# Calculate Spearman correlation between PEDM and ADM
spearman_adm_pedm = spearmanr(global_pred_euc_dist.flatten(), global_data_dm.flatten()).correlation
print("\nCorrelation between PEDM and ADM:")
print(f"Pearson Correlation: {pearson_adm_pedm:.3f}")
print(f"Spearman Correlation: {spearman_adm_pedm:.3f}")


# Printing F1 scores for different clustering methods
print("\nF1 Scores:")
print(f"F1 Score (PEDM vs True): {evaluation(global_labels_true, pedm_clustering.labels_):.3f}")
print(f"F1 Score (DBSCAN vs True): {evaluation(global_labels_true, db.labels_):.3f}")
print(f"F1 Score (PEDM vs DBSCAN): {evaluation(pedm_clustering.labels_, db.labels_):.3f}")

