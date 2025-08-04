from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ferminet import checkpoint, psiformer
from ferminet.base_config import default
from ferminet.utils.system import Atom


def build_config() -> 'default':
    """Returns a configuration for a single 2D Hydrogen-like atom."""
    cfg = default()
    cfg.system.ndim = 2
    cfg.system.electrons = (1, 0)  # One electron total
    cfg.system.molecule = [Atom("X", (0.0, 0.0))]
    cfg.pretrain.iterations = 0
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot |psi| on a 2D grid")
    parser.add_argument("--checkpoint", type=str, default="ferminet_2025_08_04_14:05:20/qmcjax_ckpt_002748.npz", help="Path to trained parameters.")
    parser.add_argument("--grid", type=int, default=40, help="Grid points per axis.")
    parser.add_argument("--extent", type=float, default=5.0, help="Grid range in bohr.")
    args = parser.parse_args()

    cfg = build_config()
    ndim = cfg.system.ndim
    nspins = cfg.system.electrons

    atoms = jnp.stack([jnp.asarray(a.coords) for a in cfg.system.molecule])
    charges = jnp.asarray([a.charge for a in cfg.system.molecule])

    net = psiformer.make_fermi_net(
        nspins=nspins, charges=charges, ndim=ndim, **cfg.network.psiformer
    )

    if args.checkpoint:
        params = checkpoint.restore(args.checkpoint)[2]
    else:
        params = net.init(jax.random.PRNGKey(0))

    # Build grid
    x = jnp.linspace(-args.extent, args.extent, args.grid)
    y = jnp.linspace(-args.extent, args.extent, args.grid)
    xx, yy = jnp.meshgrid(x, y)
    positions = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (N^2, 2)

    spins = jnp.full((positions.shape[0], 1), 1.0)  # one electron → spin shape (N^2, 1)

    @jax.jit
    def single_eval(pos: jnp.ndarray, spin: jnp.ndarray) -> jnp.ndarray:
        return net.apply(params, pos, spin, atoms, charges)[1]  # log|ψ|

    logabs = jax.vmap(single_eval)(positions, spins)
    amplitude = jnp.exp(logabs).reshape(args.grid, args.grid)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(x, y, amplitude, levels=50, cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$|\psi(x, y)|$ for 1 electron")
    plt.colorbar(label=r"$|\psi|$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
