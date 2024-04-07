import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# New inputs for Federated Euclidean Distance Matrix (fedm)
fedm = np.array([[0.4, 0.2, 0.6],
                 [0.3, 0.1, 0.8],
                 [0.7, 0.9, 0.5]])

# New inputs for Predicted Euclidean Distance Matrix (pred_euc_dist)
pred_euc_dist = np.array([[0.3, 0.7, 0.5],
                          [0.9, 0.2, 0.4],
                          [0.1, 0.6, 0.8]])

# Apply Hungarian Algorithm
cost_matrix = np.abs(fedm - pred_euc_dist)
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Update the Predicted Euclidean Distance Matrix (PEDM)
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

# Highlight the optimal assignment in black
for i, j in zip(row_indices, col_indices):
    axs[2].text(j, i, 'X', color='black', fontsize=12, ha='center', va='center')

plt.show()
