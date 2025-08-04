import numpy as np
import jax.numpy as jnp

# -------------------------------
# Load the checkpoint
# -------------------------------
ckpt = np.load("ferminet_hn_2025_06_27_10:55:12/qmcjax_ckpt_000812.npz", allow_pickle=True)

# -------------------------------
# Inspect Top-Level Keys
# -------------------------------
print("Top-level keys in checkpoint:", ckpt.files)

# -------------------------------
# Inspect Params
# -------------------------------
params = ckpt["params"].item()
print("\n--- PARAMS ---")

def inspect_tree(tree, prefix='params'):
    if isinstance(tree, dict):
        for k, v in tree.items():
            inspect_tree(v, prefix=f"{prefix}.{k}")
    elif isinstance(tree, (jnp.ndarray, np.ndarray)):
        print(f"{prefix}: shape={tree.shape}, dtype={tree.dtype}")
    else:
        print(f"{prefix}: type={type(tree)}")

inspect_tree(params)

# -------------------------------
# Inspect Data
# -------------------------------
data = ckpt["data"].item()
print("\n--- DATA ---")

for key, val in data.items():
    if isinstance(val, (np.ndarray, jnp.ndarray)):
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")
    elif isinstance(val, list):
        print(f"{key}: list of length {len(val)}, first item type={type(val[0])}")
    else:
        print(f"{key}: type={type(val)}")

# Optional: view one sample of positions
if 'positions' in data:
    print("\nSample positions[0]:\n", data['positions'][0])
