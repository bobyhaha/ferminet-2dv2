import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------
# Load checkpoint
# -------------------------------
ckpt = np.load("ferminet_2025_06_28_17:43:06/qmcjax_ckpt_000026.npz", allow_pickle=True)
data = ckpt["data"].item()

# -------------------------------
# Extract and reshape positions
# -------------------------------
positions_flat = data["positions"][0]  # shape: (n_walkers, n_electrons * ndim)
n_walkers, flat_dim = positions_flat.shape
ndim = 3
assert flat_dim % ndim == 0, "Expected positions to be compatible with 3D."
n_electrons = flat_dim // ndim
print(f"Loaded {n_walkers} walkers with {n_electrons} electrons in 3D space.")

positions = positions_flat.reshape((n_walkers, n_electrons, ndim))
all_electrons = positions.reshape(-1, ndim)  # (n_walkers * n_electrons, ndim)

# -------------------------------
# Create 3D histogram
# -------------------------------
bins = 40
bounds = [-2, 2]
range_nd = [bounds] * ndim

hist, edges = np.histogramdd(all_electrons, bins=bins, range=range_nd, density=True)

# -------------------------------
# Extract high-density voxel centers
# -------------------------------
threshold = 0.1 * np.max(hist)
mask = hist > threshold

# Get voxel centers
centers = [0.5 * (edges[d][1:] + edges[d][:-1]) for d in range(ndim)]
grid = np.meshgrid(*centers, indexing='ij')
coords = [g[mask] for g in grid]

# -------------------------------
# Plot 3D scatter
# -------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[0], coords[1], coords[2], c='blue', alpha=0.4, s=20)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Electron Density (High-Density Regions)")
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()
