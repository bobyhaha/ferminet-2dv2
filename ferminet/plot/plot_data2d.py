import numpy as np
import matplotlib.pyplot as plt
import os
# Load checkpoint
ckpt = np.load("ferminet_2025_07_31_08:06:14/qmcjax_ckpt_019510.npz", allow_pickle=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load checkpoint
files = os.listdir("ferminet_2025_07_31_08:06:14")
files = sorted(files, reverse=True)
num_files = 10
iter = 0
positions_list = []
for file in files:
  if ".npz" in file and iter < num_files:
    path = "ferminet_2025_07_31_08:06:14" + '/'+file
    ckpt = np.load(path, allow_pickle=True)
    data = ckpt["data"].item()
    positions = data["positions"]
    positions = positions[0]
    positions_list.append(positions)
    iter += 1
positions = np.concatenate(positions_list, axis=0)
# Reshape to (n_walkers, n_electrons, ndim)
positions = positions.reshape(positions.shape[0], -1, 2)  # (batch, n_electrons, 3)
  # (batch, n_electrons, 2)
all_electron_positions = positions.reshape(-1, 2)

x = all_electron_positions[:, 0]
y = all_electron_positions[:, 1]

# Create 2D histogram
bins = 300
hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

# Apply Gaussian smoothing
hist_smooth = gaussian_filter(hist, sigma=1.0)

# Set up plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "text.usetex": False,  # change to True if using LaTeX
})

# Plot
plt.figure(figsize=(5, 5), dpi=150)
plt.imshow(
    hist_smooth.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin='lower',
    cmap='Blues',
    aspect='equal'
)



# Remove ticks for cleaner look
plt.xticks([])
plt.yticks([])

# Add colorbar
cbar = plt.colorbar()
cbar.set_label(r"$n(\mathbf{r})$", fontsize=14)

plt.title("Electron Density", fontsize=16)
plt.tight_layout()
plt.show()
