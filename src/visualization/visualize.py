from matplotlib import pyplot as plt
import numpy as np


def plot_kmeans_clusters(df, label, centroids, title):
    '''
    Plots kmeans clusterings
            Parameters:
                    df (array of shape (n_samples, n_features)): the whole dataframe
                    label (array of shape (n_samples,)): the respective labels
                    centroids (array of shape (n_samples, n_features)): the centers of clusterings
                    title (str): the title for the plot

            Returns:
                    null
    '''
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.title(title)
    plt.show()