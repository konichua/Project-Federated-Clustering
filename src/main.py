from src.fedm_kmeans_pipeline import fedm_kmeans_pipeline
from src.data.generate_dataset import generate_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

###### participants features
PARTICIPANTS = 10  # 5
DATASET_DIVISION_TYPE = 'non-iid clusters' #['iid', 'non-iid points', 'non-iid clusters'] [gr0, gr1, gr2]
###### dataset features
# TOTAL_GLOBAL_SAMPLES = 750
# GLOBAL_CLUSTER_STD = 1
# GLOBAL_CLUSTER_FEATURES = 500
# CENTERS = 4
###### gold - where kmeans on global data performs at least with rand_score=0.716 +- 0.09
GOLD_CENTERS = 10
GOLD_CLUSTER_STD = 27
GOLD_CLUSTER_FEATURES = 500
GOLD_GLOBAL_SAMPLES = 750


# The adjusted Rand Index (ARI) should be interpreted as follows:
# ARI >= 0.90 excellent recovery
# 0.80 =< ARI < 0.90 good recovery
# 0.65 =< ARI < 0.80 moderate recovery
# ARI < 0.65 poor recovery


# # generate toy dataset
# global_data, global_labels_true = generate_dataset(cluster_std=GOLD_CLUSTER_STD - 14,
#                                                    n_features=GOLD_CLUSTER_FEATURES,
#                                                    samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                    centers=GOLD_CENTERS,
#                                                    type='blobs')
# rand_score = fedm_kmeans_pipeline(global_data=global_data,
#                                   dataset_division_type=DATASET_DIVISION_TYPE,
#                                   participants=PARTICIPANTS)

graph0_fedm_avg = {
    'iid': [],
    'non-iid points': [],
    'non-iid clusters': []
}
graph0_fedm_std = {
    'iid': [],
    'non-iid points': [],
    'non-iid clusters': []
}
graph0_kmeans_avg = []
graph0_kmeans_std = []
for n_participants in tqdm(range(1, 30)):
    arr_fedm_vs_true = {
        'iid': [],
        'non-iid points': [],
        'non-iid clusters': []
    }
    arr_kmeans_vs_true = []
    for n_rand in tqdm(range(20), leave=False):
        global_data, global_labels_true = generate_dataset(cluster_std=18,
                                                           n_features=GOLD_CLUSTER_FEATURES,
                                                           samples_nb=GOLD_GLOBAL_SAMPLES,
                                                           centers=GOLD_CENTERS,
                                                           type='blobs',
                                                           random_state=n_rand)
        for division_type in ['iid', 'non-iid points']:
            fedm_vs_true, kmeans_vs_true = fedm_kmeans_pipeline(global_data=global_data,
                                                                dataset_division_type=division_type,
                                                                participants=n_participants,
                                                                global_labels_true=global_labels_true,
                                                                gold_centers=GOLD_CENTERS,
                                                                random_state=n_rand)
            arr_fedm_vs_true[division_type].append(fedm_vs_true)
            arr_kmeans_vs_true.append(kmeans_vs_true)
    for key, values in arr_fedm_vs_true.items():
        graph0_fedm_avg[key].append(np.mean(values))
        graph0_fedm_std[key].append(np.std(values))
    graph0_kmeans_avg.append(np.mean(arr_kmeans_vs_true))
    graph0_kmeans_std.append(np.std(arr_kmeans_vs_true))

# Plotting
plt.figure(figsize=(10, 6))
colors = ['blue', 'black', 'green']
for color, key in zip(colors, graph0_fedm_avg):
    if len(graph0_fedm_avg[key]) == 0:
        continue
    plt.plot(range(1, 30), graph0_fedm_avg[key], label=key, color=color)
    plt.fill_between(range(1, 30), np.array(graph0_fedm_avg[key]) - np.array(graph0_fedm_std[key]),
                     np.array(graph0_fedm_avg[key]) + np.array(graph0_fedm_std[key]),
                     color='grey', alpha=0.2)
plt.title('LSDM participants')
# plt.plot(range(1, 30), graph0_fedm_avg, label='FEDM', color='blue')
# plt.plot(range(1, 30), graph0_kmeans_avg, label='K-means', color='black')
# plt.fill_between(range(1, 30), np.array(graph0_fedm_avg) - np.array(graph0_fedm_std),
#                  np.array(graph0_fedm_avg) + np.array(graph0_fedm_std), color='grey', alpha=0.2)
# plt.fill_between(range(1, 30), np.array(graph0_kmeans_avg) - np.array(graph0_kmeans_std),
#                  np.array(graph0_kmeans_avg) + np.array(graph0_kmeans_std), color='grey', alpha=0.2)
#
# # Customization
# plt.title("FEDM vs K-means")

plt.xlabel("N participants")
plt.ylabel("Adjusted Rand Score")
plt.legend()

# Show plot
plt.show()





















# graph2_participants = []
# for n_participants in [2, 3]: #[2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     adj_rand_score = fedm_kmeans_pipeline(global_data=global_data,
#                                       dataset_division_type=DATASET_DIVISION_TYPE,
#                                       participants=n_participants, global_labels_true=global_labels_true)
#     graph2_participants.append(adj_rand_score)
# print(graph2_participants)

# [0.8693409473640623, 0.8968461644401597, 0.764411964641955, 0.8968461644401597, 0.7808354326883084,
# 0.7810779188602166, 0.8968461644401597, 0.8968461644401597, 0.8968461644401597]
# adj_rand_score = fedm_kmeans_pipeline(global_data=global_data,
#                                       dataset_division_type=DATASET_DIVISION_TYPE,
#                                       participants=1, global_labels_true=global_labels_true)
# print(adj_rand_score)
