from src.visualization.visualize import plot_kmeans_clusters
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score
from src.data.generate_spikes import generate_spikes
from src.data.divide_dataset import divide_dataset
from src.data.clustering import clustering
from matplotlib import pyplot as plt
import numpy as np


def fedm_kmeans_pipeline(global_data, dataset_division_type, participants, global_labels_true, gold_centers,
                         random_state, algorithm_name):

    # plt.scatter(global_data[:, 0], global_data[:, 1])
    # plt.title('Global data')
    # plt.show()

    # global data clustering
    global_clusterer = clustering(algorithm_name, gold_centers, random_state)
    global_labels_kmeans = global_clusterer.fit_predict(global_data)
    # plot_kmeans_clusters(global_data, global_labels_kmeans, global_clusterer.cluster_centers_, 'Global data')


    # dividing global data by participants
    participants_data, participants_labels = divide_dataset(global_data, participants,
                                                            global_labels_true.copy(),
                                                            dataset_division_type, random_seed=random_state)


    #### Participant Based Computation ####
    spikes = generate_spikes(participants_data, global_data.shape[1])
    print(f'{spikes.shape=}')
    lsdm = [euclidean_distances(d, spikes) for d in participants_data]

    #### Coordinator Based Computation ####
    lsdm_concat = np.concatenate(lsdm)
    # fedm_matrix = euclidean_distances(lsdm_concat)
    fedm_model = clustering(algorithm_name, gold_centers, random_state)

    # here predict on either fedm_matrix OR lsdm_concat
    fedm_labels = fedm_model.fit_predict(lsdm_concat)

    # print('===============')
    # print(f'Pearson FEDM vs ADM: {pearson_adm_fedm[0, 1]:.3f}')
    # print(f'Spearman FEDM vs ADM: {spearman_adm_fedm.correlation:.3f}')
    return (adjusted_rand_score(participants_labels, fedm_labels),
            adjusted_rand_score(global_labels_true, global_labels_kmeans))
