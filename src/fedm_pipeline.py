from src.visualization.visualize import plot_kmeans_clusters
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score
from src.data.generate_spikes import generate_spikes
from src.data.divide_dataset import divide_dataset
from src.data.clustering import clustering
from matplotlib import pyplot as plt
import numpy as np


def fedm_pipeline(global_data, dataset_division_type, participants, global_labels_true, gold_centers,
                  random_state, algorithm_name):
    '''
    Returns the generated blobs with labels
            Parameters:
                    global_data (ndarray of shape (n_samples, n_features)): the dataset
                    dataset_division_type (str): the type of dataset partitioning
                    participants (int): the number of participants
                    global_labels_true (ndarray of shape (n_samples,)):
                        the integer labels for cluster membership of each sample
                    gold_centers (int): the numer of clusters
                    random_state (int): pass an int for reproducible output
                    algorithm_name (str): the name of the clustering algorithm

            Returns:
                    ari_lsdm (float): the Adjusted Rand Index for CLSDM federated clustering
                    ari_algo (float): the Adjusted Rand Index for the global data clustering
    '''
    # plt.scatter(global_data[:, 0], global_data[:, 1])
    # plt.title('Global data')
    # plt.show()

    # global data clustering
    global_clusterer = clustering(algorithm_name, gold_centers, random_state)
    global_labels_clusterer = global_clusterer.fit_predict(global_data)
    # plot_kmeans_clusters(global_data, global_labels_kmeans, global_clusterer.cluster_centers_, 'Global data')


    # dividing global data by participants
    participants_data, participants_labels = divide_dataset(global_data, participants,
                                                            global_labels_true.copy(),
                                                            dataset_division_type, random_seed=random_state)


    #### Participant Based Computation ####
    spikes = generate_spikes(participants_data, global_data.shape[1])
    # print(f'{spikes.shape=}')
    lsdm = [euclidean_distances(d, spikes) for d in participants_data]

    #### Coordinator Based Computation ####
    lsdm_concat = np.concatenate(lsdm)
    fedm_matrix = euclidean_distances(lsdm_concat)
    model = clustering(algorithm_name, gold_centers, random_state)

    # here predict labels for either lsdm or fedm
    lsdm_labels = model.fit_predict(lsdm_concat)
    print(global_labels_clusterer)
    fedm_labels = model.fit_predict(fedm_matrix)

    ari_lsdm = adjusted_rand_score(participants_labels, lsdm_labels)
    ari_algo = adjusted_rand_score(global_labels_true, global_labels_clusterer)
    # ari_fedm can be returned instead of ari_algo if needed
    ari_fedm = adjusted_rand_score(participants_labels, fedm_labels)
    return ari_lsdm, ari_algo
