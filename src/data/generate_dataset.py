from sklearn.datasets import make_blobs


def generate_dataset(cluster_std, samples_nb, centers, n_features, center_box=(-10.0, 10.0), random_state=0):
    '''
    Returns the generated blobs with labels
            Parameters:
                    cluster_std (int): the standard deviation of the clusters
                    samples_nb (int): the total number of points
                    centers (int): the number of centers to generate
                    n_features (int): the number of features for each sample
                    center_box (int): the bounding box for each cluster center
                    random_state (int): pass an int for reproducible output

            Returns:
                    global_df (ndarray of shape (n_samples, n_features)): the generated samples
                    global_labels_true (ndarray of shape (n_samples,)): the integer labels for cluster membership of each sample
    '''
    global_df, global_labels_true = make_blobs(n_samples=samples_nb,
                                               n_features=n_features,
                                               centers=centers,
                                               center_box=center_box,
                                               cluster_std=cluster_std,
                                               random_state=random_state)
    return global_df, global_labels_true
