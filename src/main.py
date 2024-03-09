from src.fedm_pipeline import fedm_pipeline
from src.data.generate_dataset import generate_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

###### participants features
PARTICIPANTS = 14  # 14
DATASET_DIVISION_TYPE = 'iid' #['iid', 'non-iid points', 'non-iid clusters'] [gr0, gr1, gr2]

# ['kmeans', 'meanshift', 'affinitypropagation', 'birch', 'spectralclustering', 'gaussianmixture', 'dbscan', 'optics']
# 'kmeans', 'meanshift', 'affinitypropagation', 'birch', 'gaussianmixture'
ALGORITHM_NAME = 'affinitypropagation'

###### dataset features
###### gold - where kmeans on global data performs at least with rand_score=0.7
GOLD_CENTERS = 5
GOLD_CLUSTER_STD = 10
GOLD_CLUSTER_FEATURES = 20
GOLD_GLOBAL_SAMPLES = 573



# # # =========================================== STD ==================================================================
# lsdm_avg = []
# lsdm_std = []
# algo_avg = []
# algo_std = []
# for n_std in tqdm(range(0, 15)):
#     arr_lsdm_vs_true = []
#     arr_algo_vs_true = []
#     for n_rand in tqdm(range(20), leave=False):
#         global_data, global_labels_true = generate_dataset(cluster_std=n_std,
#                                                            n_features=GOLD_CLUSTER_FEATURES,
#                                                            samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=n_rand)
#         lsdm_vs_true, algo_vs_true = fedm_pipeline(global_data=global_data,
#                                                    dataset_division_type=DATASET_DIVISION_TYPE,
#                                                    participants=PARTICIPANTS,
#                                                    global_labels_true=global_labels_true,
#                                                    gold_centers=GOLD_CENTERS,
#                                                    random_state=n_rand,
#                                                    algorithm_name=ALGORITHM_NAME)
#         arr_lsdm_vs_true.append(lsdm_vs_true)
#         arr_algo_vs_true.append(algo_vs_true)
#     lsdm_avg.append(np.mean(arr_lsdm_vs_true))
#     lsdm_std.append(np.std(arr_lsdm_vs_true))
#     algo_avg.append(np.mean(arr_algo_vs_true))
#     algo_std.append(np.std(arr_algo_vs_true))
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(0, 15), lsdm_avg, label='Federated ' + ALGORITHM_NAME, color='blue')
# plt.plot(range(0, 15), algo_avg, label='Global ' + ALGORITHM_NAME, color='black')
# plt.fill_between(range(0, 15), np.array(lsdm_avg) - np.array(lsdm_std),
#                  np.array(lsdm_avg) + np.array(lsdm_std), color='grey', alpha=0.2)
# plt.fill_between(range(0, 15), np.array(algo_avg) - np.array(algo_std),
#              np.array(algo_avg) + np.array(algo_std), color='grey', alpha=0.2)
# plt.title('Federated Clustering vs Global clustering on ' + ALGORITHM_NAME)
# plt.xlabel('STD')
# plt.ylabel('Adjusted Rand Score')
# plt.legend()
# plt.show()
# # # =========================================== STD ==================================================================



def participants_datadivision_comparison(name):
    if name == 'constant_samples':
        title = 'N participants share 573 samples'
        samples_nb = GOLD_GLOBAL_SAMPLES
    elif name == 'growing_samples':
        title = 'Every participant brings 50 new samples'
    graph0_fedm_avg = {
        'iid': [],
        'non-iid points': [],
    }
    graph0_fedm_std = {
        'iid': [],
        'non-iid points': [],
    }
    gridline = list(range(1, 120, 5))
    for n_participants in tqdm(gridline):
        if name == 'growing_samples':
            samples_nb = 50 * n_participants
        arr_fedm_vs_true = {
            'iid': [],
            'non-iid points': [],
        }
        for n_rand in tqdm(range(20), leave=False):
            global_data, global_labels_true = generate_dataset(cluster_std=6,
                                                               n_features=GOLD_CLUSTER_FEATURES,
                                                               samples_nb=samples_nb,
                                                               centers=GOLD_CENTERS,
                                                               random_state=n_rand)
            for division_type in ['iid', 'non-iid points']:
                fedm_vs_true, _ = fedm_pipeline(global_data=global_data,
                                                dataset_division_type=division_type,
                                                participants=n_participants,
                                                global_labels_true=global_labels_true,
                                                gold_centers=GOLD_CENTERS,
                                                random_state=n_rand,
                                                algorithm_name='kmeans')
                arr_fedm_vs_true[division_type].append(fedm_vs_true)
        for key, values in arr_fedm_vs_true.items():
            graph0_fedm_avg[key].append(np.mean(values))
            graph0_fedm_std[key].append(np.std(values))

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'black']
    for color, key in zip(colors, graph0_fedm_avg):
        if len(graph0_fedm_avg[key]) == 0:
            continue
        plt.plot(gridline, graph0_fedm_avg[key], label=key, color=color)
        plt.fill_between(gridline, np.array(graph0_fedm_avg[key]) - np.array(graph0_fedm_std[key]),
                         np.array(graph0_fedm_avg[key]) + np.array(graph0_fedm_std[key]),
                         color=color, alpha=0.2)
    plt.title(title)
    plt.xlabel('N participants')
    plt.ylabel('Adjusted Rand Score')
    plt.legend()
    plt.show()


# participants_datadivision_comparison('growing_samples')  # 'constant_samples' 'growing_samples'



# global_data, global_labels_true = generate_dataset(cluster_std=6,
#                                                            n_features=2,
#                                                            samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=15)
# fedm_vs_true, kmeans_vs_true = fedm_kmeans_pipeline(global_data=global_data,
#                                                                 dataset_division_type=DATASET_DIVISION_TYPE,
#                                                                 participants=100,
#                                                                 global_labels_true=global_labels_true,
#                                                                 gold_centers=GOLD_CENTERS,
#                                                                 random_state=15,
#                                                                 algorithm_name='kmeans')