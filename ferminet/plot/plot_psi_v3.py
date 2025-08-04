import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ferminet import checkpoint, psiformer
from ferminet.base_config import default
from ferminet.utils.system import Atom

def get_config_2d():
    cfg = default()
    cfg.system.electrons = (1, 1) 
    cfg.system.molecule = [Atom("X", (0.0, 0.0))]
    cfg.system.ndim = 2
    cfg.pretrain.iterations = 0
    return cfg

def main():
    start_time = time.time()
    print("Initializing config and loading model...")

    cfg = get_config_2d()
    ckpt_path = "ferminet_2025_07_31_09:49:22/qmcjax_ckpt_135248.npz"
    params = checkpoint.restore(ckpt_path)[2]

    atoms   = jnp.stack([jnp.array(a.coords) for a in cfg.system.molecule])  # shape (n_atoms, ndim)
    charges = jnp.array([a.charge for a in cfg.system.molecule]) # shape (n_atoms,)
    nspins  = cfg.system.electrons  # (1, 1)

    netcfg = cfg.network.psiformer
    network = psiformer.make_fermi_net(
        nspins=nspins,
        charges=charges,
        ndim=cfg.system.ndim,
        **netcfg,
    )
    x1, y1 = 0.0, 0.0
    fixed_spin_up = jnp.array([x1, y1])
    spin_vec = jnp.array([1.0, -1.0])
    N = 20
    grid = jnp.linspace(-2.0, 2.0, N)
    x2_grid, y2_grid = jnp.meshgrid(grid, grid)
    x2_flat = x2_grid.ravel()
    y2_flat = y2_grid.ravel()
    pos_grid = jnp.stack([
        jnp.full_like(x2_flat, x1),  
        jnp.full_like(y2_flat, y1), 
        x2_flat,
        y2_flat
    ], axis=-1)  # shape (N², 4)

    spin_grid = jnp.tile(spin_vec[None, :], (pos_grid.shape[0], 1))  # shape (N², 2)
    @jax.jit
    def single_logabs(pos, spin):
        return network.apply(params, pos, spin, atoms, charges)[1]

    batched_logabs = jax.vmap(single_logabs, in_axes=(0, 0))
    logabs = batched_logabs(pos_grid, spin_grid)
    Z = jnp.exp(logabs).reshape(N, N)

    print(f"Done in {time.time() - start_time:.2f} seconds.")
    plt.figure(figsize=(6, 5))
    plt.contourf(grid, grid, Z, levels=50, cmap="viridis")
    plt.colorbar(label=r"$|\psi|$ (e₂)")
    plt.scatter([x1], [y1], color="white", s=50, label="e₁ fixed (spin ↑)")
    plt.xlabel("Electron 2 x")
    plt.ylabel("Electron 2 y")
    plt.title(rf"2D $|\psi|$ with $e_1 = ({x1},{y1})$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("wavefunction_plot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
