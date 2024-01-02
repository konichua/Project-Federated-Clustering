from sklearn.datasets import make_blobs

default_centers = [[1, 1], [-1, -1]]
def generate_dataset(cluster_std, samples_nb, type, n_features=2, centers=default_centers, random_state=0):
    if type == 'blobs':
        global_df, global_labels_true = make_blobs(n_samples=samples_nb, n_features=n_features,
                                    centers=centers, cluster_std=cluster_std, random_state=random_state)
    else:
        raise ValueError('type not specified')
    return global_df, global_labels_true