from sklearn.datasets import make_blobs


def generate_dataset(cluster_std, samples_nb, centers, type, n_features, center_box=(-10.0, 10.0), random_state=0):
    if type == 'blobs':
        global_df, global_labels_true = make_blobs(n_samples=samples_nb,
                                                   n_features=n_features,
                                                   centers=centers,
                                                   center_box=center_box,
                                                   cluster_std=cluster_std,
                                                   random_state=random_state)
    else:
        raise ValueError('type not specified')
    return global_df, global_labels_true
