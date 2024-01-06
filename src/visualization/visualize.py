from matplotlib import pyplot as plt
import numpy as np

def plot_db_clusters(db, X, savefig_name=None, title=None):
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise_ = list(db.labels_).count(-1)
    unique_labels = set(labels)

    #     print(f'Estimated number of clusters: {n_clusters_}')
    #     print(f'Estimated number of noise points: {n_noise_}')
    #     print(f'Unique labels: {unique_labels}')

    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    if title:
        plt.title(f'{title} \n clusters: {n_clusters_}, noise points: {n_noise_}')
    else:
        plt.title(f'clusters: {n_clusters_}, noise points: {n_noise_}')
    if savefig_name:
        plt.savefig(savefig_name)
    else:
        plt.show()