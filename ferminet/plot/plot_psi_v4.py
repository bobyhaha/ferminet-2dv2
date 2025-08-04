"""Plot the magnitude of a 2D Psiformer wavefunction on a grid.

This example builds a Psiformer FermiNet with ``ndim=2`` and evaluates
``|psi|`` on a 2D grid where one electron is fixed at a chosen location while
the second electron scans the plane around it. A checkpoint with trained
parameters can be provided, otherwise the network uses randomly initialised
parameters.

Example usage:
    python -m ferminet.plot.plot_psiformer_2d --checkpoint /path/to/ckpt.npz
"""

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
    cfg.system.electrons = (2,0)
    cfg.system.molecule = [Atom("X", (0.0, 0.0))]
    cfg.pretrain.iterations = 0
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot |psi| on a 2D grid")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ferminet_2025_08_04_15:07:46/qmcjax_ckpt_010508.npz",
        help="Optional path to checkpoint containing trained parameters.",
    )
    parser.add_argument(
        "--grid", type=int, default=40, help="Number of grid points per axis."
    )
    parser.add_argument(
        "--extent", type=float, default=5.0, help="Plot range in bohr."
    )
    parser.add_argument(
        "--fixed-x", type=float, default=0.0, help="x-position of fixed electron"
    )
    parser.add_argument(
        "--fixed-y", type=float, default=0.0, help="y-position of fixed electron"
    )
    args = parser.parse_args()

    cfg = build_config()

    atoms = jnp.stack([jnp.asarray(a.coords) for a in cfg.system.molecule])
    charges = jnp.asarray([a.charge for a in cfg.system.molecule])
    nspins = cfg.system.electrons

    net = psiformer.make_fermi_net(
        nspins=nspins, charges=charges, ndim=cfg.system.ndim, **cfg.network.psiformer
    )

    if args.checkpoint:
        params = checkpoint.restore(args.checkpoint)[2]
    else:
        params = net.init(jax.random.PRNGKey(0))

    fixed_pos = jnp.array([args.fixed_x, args.fixed_y])
    spin_vec = jnp.array([5.0, -5.0])

    x_grid = jnp.linspace(-args.extent, args.extent, args.grid) + args.fixed_x
    y_grid = jnp.linspace(-args.extent, args.extent, args.grid) + args.fixed_y
    x2, y2 = jnp.meshgrid(x_grid, y_grid)
    x2_flat = x2.ravel()
    y2_flat = y2.ravel()

    positions = jnp.stack(
        [
            jnp.full_like(x2_flat, fixed_pos[0]),
            jnp.full_like(y2_flat, fixed_pos[1]),
            x2_flat,
            y2_flat,
        ],
        axis=-1,
    )

    spins = jnp.tile(spin_vec[None, :], (positions.shape[0], 1))

    @jax.jit
    def single_eval(pos: jnp.ndarray, spin: jnp.ndarray) -> jnp.ndarray:
        return net.apply(params, pos, spin, atoms, charges)[1]

    logabs = jax.vmap(single_eval)(positions, spins)
    amplitude = jnp.exp(logabs).reshape(args.grid, args.grid)

    plt.figure(figsize=(6, 5))
    plt.contourf(x_grid, y_grid, amplitude, levels=50, cmap="viridis")
    plt.scatter([fixed_pos[0]], [fixed_pos[1]], c="white", s=50)
    plt.xlabel("Electron 2 x")
    plt.ylabel("Electron 2 y")
    plt.title(r"$|\psi|$ with electron 1 fixed")
    plt.colorbar(label=r"$|\psi|")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
