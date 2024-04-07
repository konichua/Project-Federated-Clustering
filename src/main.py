from src.pipeline import pipeline
from src.data.generate_dataset import generate_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

###### participants features
PARTICIPANTS = 14
DATASET_DIVISION_TYPE = 'iid'  # ['iid', 'non-iid points', 'non-iid clusters']

# 'kmeans', 'meanshift', 'affinitypropagation', 'birch', 'gaussianmixture'
ALGORITHM_NAME = 'kmeans'

###### dataset features
###### gold - where kmeans on global data performs at least with rand_score=0.7
GOLD_CENTERS = 5
GOLD_CLUSTER_STD = 10
GOLD_CLUSTER_FEATURES = 20
GOLD_GLOBAL_SAMPLES = 573



'''
    how to use
        call function plot_algos() - ALGORITHM_NAME is a subject to change if needed
        or
        call function participants_datadivision_comparison()
        or 
        uncomment a simple example below
'''


def plot_algos():
    '''
    Plots FC-Algorithm performance on STD-ARI scale
    '''
    # ari_lsdm, ari_fedm, ari_pedm, ari_algo
    lsdm_avg = []
    lsdm_std = []
    fedm_avg = []
    fedm_std = []
    pedm_avg = []
    pedm_std = []
    algo_avg = []
    algo_std = []
    for n_std in tqdm(range(0, 15)):
        arr_lsdm_vs_true = []
        arr_fedm_vs_true = []
        arr_pedm_vs_true = []
        arr_algo_vs_true = []
        for n_rand in tqdm(range(20), leave=False):
            global_data, global_labels_true = generate_dataset(cluster_std=n_std,
                                                               n_features=GOLD_CLUSTER_FEATURES,
                                                               samples_nb=GOLD_GLOBAL_SAMPLES,
                                                               centers=GOLD_CENTERS,
                                                               random_state=n_rand)
            lsdm_vs_true, fedm_vs_true, pedm_vs_true, algo_vs_true = pipeline(global_data=global_data,
                                                                              dataset_division_type=DATASET_DIVISION_TYPE,
                                                                              participants=PARTICIPANTS,
                                                                              global_labels_true=global_labels_true,
                                                                              gold_centers=GOLD_CENTERS,
                                                                              random_state=n_rand,
                                                                              algorithm_name=ALGORITHM_NAME)
            arr_lsdm_vs_true.append(lsdm_vs_true)
            arr_fedm_vs_true.append(fedm_vs_true)
            arr_pedm_vs_true.append(pedm_vs_true)
            arr_algo_vs_true.append(algo_vs_true)
        lsdm_avg.append(np.mean(arr_lsdm_vs_true))
        lsdm_std.append(np.std(arr_lsdm_vs_true))

        fedm_avg.append(np.mean(arr_fedm_vs_true))
        fedm_std.append(np.std(arr_fedm_vs_true))

        pedm_avg.append(np.mean(arr_pedm_vs_true))
        pedm_std.append(np.std(arr_pedm_vs_true))

        algo_avg.append(np.mean(arr_algo_vs_true))
        algo_std.append(np.std(arr_algo_vs_true))

    plt.figure(figsize=(10, 6))
    plt.plot(range(0, 15), lsdm_avg, label='CLSDM', color='royalblue', linewidth=4)
    plt.plot(range(0, 15), fedm_avg, label='FEDM', color='darkviolet', linewidth=4)
    plt.plot(range(0, 15), pedm_avg, label='PEDM', color='seagreen', linewidth=4)
    plt.plot(range(0, 15), algo_avg, label='Global clustering', color='black', linewidth=4)
    plt.fill_between(range(0, 15), np.array(lsdm_avg) - np.array(lsdm_std),
                     np.array(lsdm_avg) + np.array(lsdm_std), color='grey', alpha=0.2)
    plt.fill_between(range(0, 15), np.array(fedm_avg) - np.array(fedm_std),
                 np.array(fedm_avg) + np.array(fedm_std), color='grey', alpha=0.2)
    plt.fill_between(range(0, 15), np.array(pedm_avg) - np.array(pedm_std),
                     np.array(pedm_avg) + np.array(pedm_std), color='grey', alpha=0.2)
    plt.fill_between(range(0, 15), np.array(algo_avg) - np.array(algo_std),
                 np.array(algo_avg) + np.array(algo_std), color='grey', alpha=0.2)
    plt.title('Clustering with ' + ALGORITHM_NAME)
    plt.xlabel('STD')
    plt.ylabel('Adjusted Rand Score')
    plt.legend()
    plt.grid()
    plt.show()



def participants_datadivision_comparison(name):
    '''
    Plots graph of N participant vs ARI
    '''
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
                _, fedm_vs_true, _, _ = pipeline(global_data=global_data,
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
    plt.grid()
    plt.show()


# # a simple example
# global_data, global_labels_true = generate_dataset(cluster_std=2,
#                                                            n_features=GOLD_CLUSTER_FEATURES,
#                                                            samples_nb=GOLD_GLOBAL_SAMPLES,
#                                                            centers=GOLD_CENTERS,
#                                                            random_state=15)
# ari_lsdm, ari_fedm, ari_pedm, ari_algo = pipeline(global_data=global_data,
#                                     dataset_division_type=DATASET_DIVISION_TYPE,
#                                     participants=PARTICIPANTS,
#                                     global_labels_true=global_labels_true,
#                                     gold_centers=GOLD_CENTERS,
#                                     random_state=15,
#                                     algorithm_name='kmeans')
# print(f'{ari_lsdm=}')
# print(f'{ari_fedm=}')
# print(f'{ari_pedm=}')
# print(f'{ari_algo=}')


# participants_datadivision_comparison('growing_samples')  # 'constant_samples' 'growing_samples'
# plot_algos()