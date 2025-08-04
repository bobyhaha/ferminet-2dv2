#!/usr/bin/env python3
# plot_cooper_pair.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# ─── Patch out the DeepMind 1‑electron mean‑pool bug ──────────────────────────────
import ferminet.networks as _netmod
_orig_csf = _netmod.construct_symmetric_features
def _patched_csf(
    h_one: jnp.ndarray,
    h_two: jnp.ndarray,
    nspins: Tuple[int, int],
    h_aux: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    # flatten any stray singleton dims before calling original
    if h_one.ndim > 2:
        h_one = h_one.reshape(h_one.shape[0], -1)
    if h_two.ndim > 3:
        h_two = h_two.reshape(h_two.shape[0], h_two.shape[1], -1)
    if h_aux is not None and h_aux.ndim > 2:
        h_aux = h_aux.reshape(h_aux.shape[0], -1)
    return _orig_csf(h_one, h_two, nspins, h_aux)
_netmod.construct_symmetric_features = _patched_csf
# ────────────────────────────────────────────────────────────────────────────────

from ferminet import checkpoint, networks, envelopes
from ferminet.configs import cooper_pair

def main():
    # 1) Load the 2D Cooper‑pair config & checkpoint
    cfg    = cooper_pair.get_config()
    ckpt   = "ferminet_2025_07_22_07:06:43/qmcjax_ckpt_000739.npz"  # adjust as needed
    params = checkpoint.restore(ckpt)[2]

    print("2D Cooper pair: full_det =", cfg.network.full_det)
    print("ndim =", cfg.system.ndim, " electrons =", cfg.system.electrons)

    # 2) Build un‑batched system arrays
    atoms   = jnp.stack([jnp.array(a.coords) for a in cfg.system.molecule])  # (natom, 2)
    charges = jnp.array([a.charge for a in cfg.system.molecule])            # (natom,)
    nspins  = cfg.system.electrons                                           # (1,1)

    # 3) Make feature‑layer & envelope
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=nspins,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get("rescale_inputs", False),
    )
    env    = envelopes.make_isotropic_envelope()
    netcfg = cfg.network.ferminet   # <-- use the same block you trained with

    # 4) Instantiate the exact 2D FermiNet you trained (via make_fermi_net, not _2d)
    network = networks.make_fermi_net(
        nspins=nspins,
        charges=charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=env,
        feature_layer=feature_layer,
        jastrow=cfg.network.get("jastrow", "default"),
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        rescale_inputs=cfg.network.get("rescale_inputs", False),
        complex_output=False,
        **netcfg,
    )

    # 5) JIT single‑walker log|ψ| evaluator
    @jax.jit
    def single_logabs(params, pos, spins):
        _, logabs = network.apply(params, pos, spins, atoms, charges)
        return logabs

    # 6) Build a 1D cut: sweep electron 1’s x from −3 → +3, fix y=0, electron 2 at origin
    grid = jnp.linspace(-3.0, 3.0, 100)
    pos_list = []
    for x in grid:
        # [e1_x, e1_y,  e2_x, e2_y]
        pos_list.append(jnp.array([ x, 0.0,   0.0, 0.0 ]))
    pos_batch = jnp.stack(pos_list, axis=0)  # shape (100, 4)
    spins     = jnp.array([1.0, -1.0])       # (2,) up/down

    # 7) Loop in Python to avoid any hidden batch‐dims
    logabs_list = []
    for i in range(pos_batch.shape[0]):
        la = single_logabs(params, pos_batch[i], spins)
        logabs_list.append(la)
    logabs_vals = jnp.stack(logabs_list, axis=0)  # (100,)

    # 8) Plot |ψ| vs x
    plt.plot(grid, jnp.exp(logabs_vals), lw=2)
    plt.xlabel("electron 1 x coordinate (a.u.)")
    plt.ylabel("|ψ|(r,0; other at 0)")
    plt.title("2D Cooper‐pair wavefunction cut")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
