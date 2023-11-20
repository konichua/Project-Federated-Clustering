import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Assuming fedm and pred_euc_dist are your Federated Euclidean Distance Matrix and Predicted Distance Matrix
# fedm and pred_euc_dist should be square matrices of the same size

# Step 2: Calculate Pairwise Euclidean Distances
fedm = np.array([[0.5, 0.8, 0.2],
                 [0.3, 0.9, 0.6],
                 [0.7, 0.4, 0.1]])

# Example of predicted distances (replace with your actual predicted distances)
pred_euc_dist = np.array([[0.4, 0.9, 0.1],
                          [0.2, 0.8, 0.5],
                          [0.6, 0.3, 0.2]])

# Step 3: Apply Hungarian Algorithm
cost_matrix = np.abs(fedm - pred_euc_dist)
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Step 4: Update the Predicted Euclidean Distance Matrix
pedm = pred_euc_dist[row_indices, :][:, col_indices]

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot Federated Euclidean Distance Matrix
axs[0].imshow(fedm, cmap='viridis', interpolation='nearest')
axs[0].set_title('Federated Euclidean Distance Matrix')

# Plot Predicted Euclidean Distance Matrix
axs[1].imshow(pred_euc_dist, cmap='viridis', interpolation='nearest')
axs[1].set_title('Predicted Euclidean Distance Matrix')

# Plot Updated Predicted Euclidean Distance Matrix (PEDM)
axs[2].imshow(pedm, cmap='viridis', interpolation='nearest')
axs[2].set_title('Updated Predicted Euclidean Distance Matrix (PEDM)')

# Highlight the optimal assignment in red
for i, j in zip(row_indices, col_indices):
    axs[2].text(j, i, 'X', color='red', fontsize=12, ha='center', va='center')

plt.show()
