import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Function for linear regression
def perform_regression(lsdm_prime, ldm):
    model = LinearRegression()
    model.fit(lsdm_prime, ldm)
    slope = model.coef_
    intercept = model.intercept_
    return slope, intercept

# Function for coefficients with the coordinator
def share_coefficients(participant_coefficients):
    mean_coefficients = np.mean(participant_coefficients, axis=0)
    return mean_coefficients

#  construct PEDM using shared coefficients
def construct_pedm(participant_coefficients, lsdm_prime):
    mean_coefficients = share_coefficients(participant_coefficients)
    pedm = np.dot(lsdm_prime, mean_coefficients[0]) + mean_coefficients[1]
    return pedm

# Generate toy dataset
np.random.seed(30)
D1, true_labels_D1 = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=0.5, random_state=30)
D2, true_labels_D2 = make_blobs(n_samples=100, centers=[[3, 3]], cluster_std=0.5, random_state=30)

#  Affinity Propagation clustering
centroids_D1 = AffinityPropagation().fit(D1).cluster_centers_
centroids_D2 = AffinityPropagation().fit(D2).cluster_centers_

# Generate spikes 
spikes_D1 = np.concatenate([centroids_D1] * len(D1))
spikes_D2 = np.concatenate([centroids_D2] * len(D2))

# Compute local distance matrices
lsdm_D1 = euclidean_distances(D1, spikes_D1)
lsdm_D2 = euclidean_distances(D2, spikes_D2)

# Concatenate local distance matrices
LSDM_concat = np.concatenate([lsdm_D1, lsdm_D2])

# linear regression for each participant
coefficients_D1 = perform_regression(lsdm_D1, LSDM_concat)
coefficients_D2 = perform_regression(lsdm_D2, LSDM_concat)

# coefficients with the coordinator
participant_coefficients = [coefficients_D1, coefficients_D2]

# Construct PEDM using coefficients
pedm_D1 = construct_pedm(participant_coefficients, lsdm_D1)
pedm_D2 = construct_pedm(participant_coefficients, lsdm_D2)

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(D1[:, 0], D1[:, 1], label='D1')
plt.scatter(D2[:, 0], D2[:, 1], label='D2')
plt.title('Original Dataset')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(spikes_D1[:, 0], spikes_D1[:, 1], label='Spikes D1')
plt.scatter(spikes_D2[:, 0], spikes_D2[:, 1], label='Spikes D2')
plt.title('Spikes')

plt.subplot(1, 3, 3)
plt.imshow(np.vstack([pedm_D1, pedm_D2]), cmap='viridis', aspect='auto')
plt.title('Predicted Euclidean Distance Matrix (PEDM)')

plt.tight_layout()
plt.show()
