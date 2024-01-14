import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from yellowbrick.cluster import KElbowVisualizer


MAX_CLUSTERS = 20


def get_number_of_clusters(df):
    model = KMeans(n_init='auto', random_state=10)
    #  calinski_harabasz 6 distortion 12 silhouette 9
    visualizer = KElbowVisualizer(model, k=(1, MAX_CLUSTERS), metric='distortion', timings=False)
    visualizer.fit(df)  # Fit the data to the visualizer
    # visualizer.show()  # Finalize and render the figure
    # print(visualizer.elbow_value_)
    k = visualizer.elbow_value_ or 1
    score = visualizer.elbow_score_
    return k, score


def get_number_of_participants_clusters(participants_df):
    # used only in spikes generation
    k = []
    elbow_score = []
    for df in participants_df:
        k_value, score = get_number_of_clusters(df)
        k.append(k_value)
        elbow_score.append(np.round(score, 3))
    # print(f'Participants elbow_score:{elbow_score:}')
    return k

# def get_number_of_clusters(data, min_clusters=False):
#     range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     visualize_silhoulette(data)
#     confidence_rate = 0.2  # the procent of samles in one cluster to be greater than silhouette_avg
#     result = []
#     for n_clusters in range_n_clusters:
#         clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
#         cluster_labels = clusterer.fit_predict(data)
#         silhouette_avg = silhouette_score(data, cluster_labels)
#         # print(f'For n_clusters ={n_clusters} The average silhouette_score is :{silhouette_avg}')
#         sample_silhouette_values = silhouette_samples(data, cluster_labels)
#         for i in range(n_clusters):
#             # Aggregate the silhouette scores for samples belonging to cluster i
#             ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#             # amount 20%
#             confidence_threshold = np.ceil((len(ith_cluster_silhouette_values) - 1) * confidence_rate) or 1
#             good_samples = ith_cluster_silhouette_values[ith_cluster_silhouette_values > silhouette_avg]
#             # print(f'larger_then_average:{larger_then_silhouette_avg}')
#             # print(f'confidence_threshold:{confidence_threshold}')
#             if len(good_samples) < confidence_threshold:
#                 break
#             if i == n_clusters - 1:
#                 result.append(n_clusters)
#     # clusters aren't distinguished
#     print(f'possible clusters:{result}')
#     if not result:
#         return 1
#     # favors min number of clusters
#     if min_clusters:
#         return min(result)
#     # favors max number of clusters
#     return max(result)


# def visualize_silhoulette(X):
#     range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     for n_clusters in range_n_clusters:
#         # Create a subplot with 1 row and 2 columns
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         fig.set_size_inches(18, 7)
#
#         # The 1st subplot is the silhouette plot
#         # The silhouette coefficient can range from -1, 1 but in this example all
#         # lie within [-0.1, 1]
#         ax1.set_xlim([-0.1, 1])
#         # The (n_clusters+1)*10 is for inserting blank space between silhouette
#         # plots of individual clusters, to demarcate them clearly.
#         ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#         # Initialize the clusterer with n_clusters value and a random generator
#         # seed of 10 for reproducibility.
#         clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
#         cluster_labels = clusterer.fit_predict(X)
#
#         # The silhouette_score gives the average value for all the samples.
#         # This gives a perspective into the density and separation of the formed
#         # clusters
#         silhouette_avg = silhouette_score(X, cluster_labels)
#         print(
#             "For n_clusters =",
#             n_clusters,
#             "The average silhouette_score is :",
#             silhouette_avg,
#         )
#
#         # Compute the silhouette scores for each sample
#         sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#         y_lower = 10
#         for i in range(n_clusters):
#             # Aggregate the silhouette scores for samples belonging to
#             # cluster i, and sort them
#             ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#
#             ith_cluster_silhouette_values.sort()
#
#             size_cluster_i = ith_cluster_silhouette_values.shape[0]
#             y_upper = y_lower + size_cluster_i
#
#             color = cm.nipy_spectral(float(i) / n_clusters)
#             ax1.fill_betweenx(
#                 np.arange(y_lower, y_upper),
#                 0,
#                 ith_cluster_silhouette_values,
#                 facecolor=color,
#                 edgecolor=color,
#                 alpha=0.7,
#             )
#
#             # Label the silhouette plots with their cluster numbers at the middle
#             ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#             # Compute the new y_lower for next plot
#             y_lower = y_upper + 10  # 10 for the 0 samples
#
#         ax1.set_title("The silhouette plot for the various clusters.")
#         ax1.set_xlabel("The silhouette coefficient values")
#         ax1.set_ylabel("Cluster label")
#
#         # The vertical line for average silhouette score of all the values
#         ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#         ax1.set_yticks([])  # Clear the yaxis labels / ticks
#         ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#         # 2nd Plot showing the actual clusters formed
#         colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#         ax2.scatter(
#             X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
#         )
#
#         # Labeling the clusters
#         centers = clusterer.cluster_centers_
#         # Draw white circles at cluster centers
#         ax2.scatter(
#             centers[:, 0],
#             centers[:, 1],
#             marker="o",
#             c="white",
#             alpha=1,
#             s=200,
#             edgecolor="k",
#         )
#
#         for i, c in enumerate(centers):
#             ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
#
#         ax2.set_title("The visualization of the clustered data.")
#         ax2.set_xlabel("Feature space for the 1st feature")
#         ax2.set_ylabel("Feature space for the 2nd feature")
#
#         plt.suptitle(
#             "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
#             % n_clusters,
#             fontsize=14,
#             fontweight="bold",
#         )
#     plt.show()

# def get_number_of_clusters(data, min_clusters=False):
#     range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     silhouette_values = []
#     threshold = 0.6
#     for n_clusters in range_n_clusters:
#         clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
#         cluster_labels = clusterer.fit_predict(data)
#         silhouette_avg = silhouette_score(data, cluster_labels)
#         silhouette_values.append(silhouette_avg)
#     silhouette_values = np.asarray(silhouette_values)
#     # print(f'silhouette_values:{silhouette_values}')
#     if any(silhouette_values > threshold):
#         # favors min number of clusters
#         if min_clusters:
#             return np.argmax(silhouette_values > threshold) + range_n_clusters[0]
#         # favors max number of clusters
#         return np.argmax(silhouette_values) + range_n_clusters[0]
#     return 1


# def get_number_of_participants_clusters(participants_data, min_clusters=False):
#     range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     silhouette_values = [[] for _ in range(participants_data.shape[0])]
#     threshold = 0.6
#     participant_nb = 0
#     result = []
#     for data in participants_data:
#         for n_clusters in range_n_clusters:
#             clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
#             cluster_labels = clusterer.fit_predict(data)
#
#             # The silhouette_score gives the average value for all the samples.
#             # This gives a perspective into the density and separation of the formed clusters
#             silhouette_avg = silhouette_score(data, cluster_labels)
#             silhouette_values[participant_nb].append(silhouette_avg)
#             # print(f'For participant_nb = {participant_nb} For n_clusters = {n_clusters} '
#             #       f'The average silhouette_score={silhouette_avg}')
#         participant_nb += 1
#     silhouette_values = np.asarray(silhouette_values)
#     # print(f'silhouette_values:{silhouette_values}')
#     for participant_values in silhouette_values:
#         # first occurrence of value greater than threshold
#         # get minimum number of clusters
#         if any(participant_values > threshold):
#             # favors min number of clusters
#             if min_clusters:
#                 result.append(np.argmax(participant_values > threshold) + range_n_clusters[0])
#             # favors max number of clusters
#             else:
#                 result.append(np.argmax(participant_values) + range_n_clusters[0])
#         else:
#             result.append(-1)
#     return result



# from sklearn.datasets import make_blobs
# from yellowbrick.cluster import KElbowVisualizer
#
# # Generate synthetic dataset with 8 random clusters
# X, y = make_blobs(n_samples=750, n_features=20, cluster_std=10, centers=None, random_state=10)
#
# # Instantiate the clustering model and visualizer
# model = KMeans(n_init='auto')
# visualizer = KElbowVisualizer(
#     model, k=(2,12), metric='distortion', timings=False  # calinski_harabasz 6 distortion 12 silhouette 9
# )
#
# visualizer.fit(X)        # Fit the data to the visualizer
# print(visualizer.elbow_value_)
# visualizer.show()        # Finalize and render the figure