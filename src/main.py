from src.fedm_pipeline import fedm_kmeans_pipeline
from src.data.generate_dataset import generate_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#
###### participants features
PARTICIPANTS = 14  # 14
DATASET_DIVISION_TYPE = 'iid' #['iid', 'non-iid points', 'non-iid clusters'] [gr0, gr1, gr2]
###### dataset features
# TOTAL_GLOBAL_SAMPLES = 750
# GLOBAL_CLUSTER_STD = 1
# GLOBAL_CLUSTER_FEATURES = 500
# CENTERS = 4
# ALGORITHM_NAME = 'kmeans'
# ['kmeans', 'meanshift', 'affinitypropagation', 'birch', 'spectralclustering', 'gaussianmixture', 'dbscan', 'optics'] spectralclustering?
ALGORITHM_NAMES = ['kmeans'] #, 'kmeans', 'meanshift', 'affinitypropagation', 'birch', 'gaussianmixture']
###### gold - where kmeans on global data performs at least with rand_score=0.7
GOLD_CENTERS = 5
GOLD_CLUSTER_STD = 10
GOLD_CLUSTER_FEATURES = 20
GOLD_GLOBAL_SAMPLES = 573



# # =========================================== STD ==================================================================
# # ['kmeans', 'meanshift', 'affinitypropagation', 'birch', 'spectralclustering', 'gaussianmixture', 'dbscan', 'optics']
# lsdm_avg = {
#     'kmeans': [],
#     'meanshift': [],
#     'affinitypropagation': [],
#     'birch': [],
#     'spectralclustering': [],
#     'gaussianmixture': [],
#     'dbscan': [],
#     'optics': []
# }
# lsdm_std = {
#     'kmeans': [],
#     'meanshift': [],
#     'affinitypropagation': [],
#     'birch': [],
#     'spectralclustering': [],
#     'gaussianmixture': [],
#     'dbscan': [],
#     'optics': []
# }
# algo_avg = {
#     'kmeans': [],
#     'meanshift': [],
#     'affinitypropagation': [],
#     'birch': [],
#     'spectralclustering': [],
#     'gaussianmixture': [],
#     'dbscan': [],
#     'optics': []
# }
# algo_std = {
#     'kmeans': [],
#     'meanshift': [],
#     'affinitypropagation': [],
#     'birch': [],
#     'spectralclustering': [],
#     'gaussianmixture': [],
#     'dbscan': [],
#     'optics': []
# }
# for n_std in tqdm(range(0, 15)):
#     arr_lsdm_vs_true = {
#         'kmeans': [],
#         'meanshift': [],
#         'affinitypropagation': [],
#         'birch': [],
#         'spectralclustering': [],
#         'gaussianmixture': [],
#         'dbscan': [],
#         'optics': []
#     }
#     arr_algo_vs_true = {
#         'kmeans': [],
#         'meanshift': [],
#         'affinitypropagation': [],
#         'birch': [],
#         'spectralclustering': [],
#         'gaussianmixture': [],
#         'dbscan': [],
#         'optics': []
#     }
#     for n_rand in tqdm(range(20), leave=False):
#         global_data, global_labels_true = generate_dataset(cluster_std=n_std,
#                                                            n_features=GOLD_CLUSTER_FEATURES,
#                                                            samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=n_rand)
#         # ['kmeans', 'meanshift', 'affinitypropagation', 'birch', 'spectralclustering', 'gaussianmixture', 'dbscan', 'optics']
#         for algo_type in ALGORITHM_NAMES:
#             lsdm_vs_true, algo_vs_true = fedm_kmeans_pipeline(global_data=global_data,
#                                                                 dataset_division_type=DATASET_DIVISION_TYPE,
#                                                                 participants=PARTICIPANTS,
#                                                                 global_labels_true=global_labels_true,
#                                                                 gold_centers=GOLD_CENTERS,
#                                                                 random_state=n_rand,
#                                                                 algorithm_name=algo_type)
#             arr_lsdm_vs_true[algo_type].append(lsdm_vs_true)
#             arr_algo_vs_true[algo_type].append(algo_vs_true)
#     for key, values in arr_lsdm_vs_true.items():
#         lsdm_avg[key].append(np.mean(values))
#         lsdm_std[key].append(np.std(values))
#     for key, values in arr_algo_vs_true.items():
#         algo_avg[key].append(np.mean(values))
#         algo_std[key].append(np.std(values))
#
# plt.figure(figsize=(10, 6))
# algo_type = ALGORITHM_NAMES[0]
# plt.plot(range(0, 15), lsdm_avg[algo_type], label='Federated ' + algo_type, color='blue')
# plt.plot(range(0, 15), algo_avg[algo_type], label='Global ' + algo_type, color='black')
# plt.fill_between(range(0, 15), np.array(lsdm_avg[algo_type]) - np.array(lsdm_std[algo_type]),
#                  np.array(lsdm_avg[algo_type]) + np.array(lsdm_std[algo_type]), color='grey', alpha=0.2)
# plt.fill_between(range(0, 15), np.array(algo_avg[algo_type]) - np.array(algo_std[algo_type]),
#              np.array(algo_avg[algo_type]) + np.array(algo_std[algo_type]), color='grey', alpha=0.2)
# plt.title('Federated Clustering vs Global clustering')
# plt.xlabel('STD')
# plt.ylabel('Adjusted Rand Score')
# plt.legend()
# plt.show()
# # =========================================== STD ==================================================================



# # =========================================== PARTICIPANTS =========================================================
# graph0_fedm_avg = {
#     'iid': [],
#     'non-iid points': [],
#     'non-iid clusters': []
# }
# graph0_fedm_std = {
#     'iid': [],
#     'non-iid points': [],
#     'non-iid clusters': []
# }
# graph0_kmeans_avg = []
# graph0_kmeans_std = []
# x_axis = list(range(1, 60, 5))
# for n_participants in tqdm(x_axis):
#     arr_fedm_vs_true = {
#         'iid': [],
#         'non-iid points': [],
#         'non-iid clusters': []
#     }
#     arr_kmeans_vs_true = []
#     for n_rand in tqdm(range(20), leave=False):
#         global_data, global_labels_true = generate_dataset(cluster_std=9,
#                                                            n_features=GOLD_CLUSTER_FEATURES,
#                                                            samples_nb=50*n_participants,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=n_rand)
#         for division_type in ['iid', 'non-iid points']:
#             fedm_vs_true, kmeans_vs_true = fedm_kmeans_pipeline(global_data=global_data,
#                                                                 dataset_division_type=division_type,
#                                                                 participants=n_participants,
#                                                                 global_labels_true=global_labels_true,
#                                                                 gold_centers=GOLD_CENTERS,
#                                                                 random_state=n_rand,
#                                                                 algorithm_name='kmeans')
#             arr_fedm_vs_true[division_type].append(fedm_vs_true)
#             arr_kmeans_vs_true.append(kmeans_vs_true)
#     for key, values in arr_fedm_vs_true.items():
#         graph0_fedm_avg[key].append(np.mean(values))
#         graph0_fedm_std[key].append(np.std(values))
#     graph0_kmeans_avg.append(np.mean(arr_kmeans_vs_true))
#     graph0_kmeans_std.append(np.std(arr_kmeans_vs_true))
#
# # Plotting
# plt.figure(figsize=(10, 6))
# colors = ['blue', 'black']
# for color, key in zip(colors, graph0_fedm_avg):
#     if len(graph0_fedm_avg[key]) == 0:
#         continue
#     plt.plot(x_axis, graph0_fedm_avg[key], label=key, color=color)
#     plt.fill_between(x_axis, np.array(graph0_fedm_avg[key]) - np.array(graph0_fedm_std[key]),
#                      np.array(graph0_fedm_avg[key]) + np.array(graph0_fedm_std[key]),
#                      color='grey', alpha=0.2)
# plt.title('K-Means Federated Clustering')
# plt.xlabel('N participants')
# plt.ylabel('Adjusted Rand Score')
# plt.legend()
# plt.show()
# =========================================== PARTICIPANTS =========================================================


# # =========================================== #Participants vs #Samples =============================================
# graph0_fedm_avg = {
#     'iid': [],
#     'non-iid points': [],
# }
# graph0_fedm_std = {
#     'iid': [],
#     'non-iid points': [],
# }
# gridline = range(1, 120, 5)
# for n_participants in tqdm(gridline):
#     arr_fedm_vs_true = {
#         'iid': [],
#         'non-iid points': [],
#     }
#     arr_kmeans_vs_true = []
#     for n_rand in tqdm(range(20), leave=False):
#         global_data, global_labels_true = generate_dataset(cluster_std=6,
#                                                            n_features=GOLD_CLUSTER_FEATURES,
#                                                            samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=n_rand)
#         for division_type in ['iid', 'non-iid points']:
#             fedm_vs_true, kmeans_vs_true = fedm_kmeans_pipeline(global_data=global_data,
#                                                                 dataset_division_type=division_type,
#                                                                 participants=n_participants,
#                                                                 global_labels_true=global_labels_true,
#                                                                 gold_centers=GOLD_CENTERS,
#                                                                 random_state=n_rand,
#                                                                 algorithm_name='kmeans')
#             arr_fedm_vs_true[division_type].append(fedm_vs_true)
#     for key, values in arr_fedm_vs_true.items():
#         graph0_fedm_avg[key].append(np.mean(values))
#         graph0_fedm_std[key].append(np.std(values))
#
# # Plotting
# plt.figure(figsize=(10, 6))
# colors = ['blue', 'black']
# for color, key in zip(colors, graph0_fedm_avg):
#     plt.plot(gridline, graph0_fedm_avg[key], label=key, color=color)
#     plt.fill_between(gridline, np.array(graph0_fedm_avg[key]) - np.array(graph0_fedm_std[key]),
#                      np.array(graph0_fedm_avg[key]) + np.array(graph0_fedm_std[key]),
#                      color=color, alpha=0.2)
# plt.title('#Participants vs #Samples')
# plt.xlabel('N participants')
# plt.ylabel('Adjusted Rand Score')
# plt.legend()
# plt.show()
# # =========================================== #Participants vs #Samples =============================================

global_data, global_labels_true = generate_dataset(cluster_std=6,
                                                           n_features=2,
                                                           samples_nb=GOLD_GLOBAL_SAMPLES,
                                                           centers=GOLD_CENTERS,
                                                           random_state=15)
fedm_vs_true, kmeans_vs_true = fedm_kmeans_pipeline(global_data=global_data,
                                                                dataset_division_type=DATASET_DIVISION_TYPE,
                                                                participants=100,
                                                                global_labels_true=global_labels_true,
                                                                gold_centers=GOLD_CENTERS,
                                                                random_state=15,
                                                                algorithm_name='kmeans')