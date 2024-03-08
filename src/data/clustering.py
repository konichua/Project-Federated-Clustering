from sklearn.cluster import (MeanShift, AffinityPropagation, KMeans,
                             SpectralClustering, Birch, DBSCAN, OPTICS)
from sklearn.mixture import GaussianMixture


def clustering(algorithm, global_clusters_nb, random_state):
    if algorithm == 'kmeans':                    # 00:03:00
        model = KMeans(n_clusters=global_clusters_nb, n_init=5, init='k-means++', random_state=random_state)
    elif algorithm == 'meanshift':               # 01:30:00
        model = MeanShift()
    elif algorithm == 'affinitypropagation':     # 00:33:00
        model = AffinityPropagation(affinity='euclidean', random_state=random_state)
    elif algorithm == 'spectralclustering':
        model = SpectralClustering(n_clusters=global_clusters_nb, assign_labels='discretize', random_state=random_state)
    elif algorithm == 'birch':                   # 00:05:00
        model = Birch(n_clusters=global_clusters_nb)
    elif algorithm == 'gaussianmixture':         # 00:28:00
        model = GaussianMixture(n_components=global_clusters_nb, init_params='k-means++', random_state=random_state)
    elif algorithm == 'dbscan':
        model = DBSCAN()
    elif algorithm == 'optics':
        model = OPTICS()
    return model