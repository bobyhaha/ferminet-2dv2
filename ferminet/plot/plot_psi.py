#!/usr/bin/env python3
# plot_psi.py
# h_ones (Array(1, dtype=int32), Array(256, dtype=int32)) 
# h_twos (Array(1, dtype=int32), Array(2, dtype=int32), Array(32, dtype=int32)) 
# g_ones (Array(2, dtype=int32), Array(256, dtype=int32)) 
# g_twos (Array(2, dtype=int32), Array(32, dtype=int32))

#!/usr/bin/env python3
# plot_psi.py

import jax
import jax.numpy as jnp
from ferminet import checkpoint, networks, envelopes
from ferminet.configs import cooper_pair

def main():
    # 1) Load config & checkpoint
    cfg = cooper_pair.get_config()
    print("full det:", cfg.network.full_det)
    print("cfg ndim:", cfg.system.ndim)

    ckpt = "ferminet_2025_07_22_07:06:43/qmcjax_ckpt_000739.npz"
    params = checkpoint.restore(ckpt)[2]

    # 2) Build un‑batched system arrays
    atoms   = jnp.stack([jnp.array(a.coords) for a in cfg.system.molecule])  # (natom, ndim)
    charges = jnp.array([a.charge for a in cfg.system.molecule])            # (natom,)
    nspins  = cfg.system.electrons                                           # e.g. (1,1)

    # 3) Build feature layer & envelope exactly as in training
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=nspins,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get("rescale_inputs", False),
    )
    env       = envelopes.make_isotropic_envelope()
    net_cfg   = cfg.network.ferminet        # ← the 3D‐constructor config

    # 4) Instantiate the **same** FermiNet you trained
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
        **net_cfg,
    )

    # 5) JIT‑compile a single‑walker log|ψ| evaluator
    @jax.jit
    def single_logabs(params, pos, spins):
        _, logabs = network.apply(params, pos, spins, atoms, charges)
        return logabs

    # 6) Prepare a batch of walker configs
    pos0   = jnp.array([1.0, 1.0, 1.0, -1.0])  # (nelec*ndim,) = (4,)
    spins0 = jnp.array([1.0, -1.0])            # (nelec,)       = (2,)
    B = 256

    pos_batch   = jnp.repeat(pos0[None],   B, axis=0)  # (B,4)
    spins_batch = jnp.repeat(spins0[None], B, axis=0)  # (B,2)

    # 7) Loop in Python to avoid any hidden vmaps
    logabs_list = []
    for i in range(B):
        la = single_logabs(params, pos_batch[i], spins_batch[i])
        logabs_list.append(la)
    logprob = 2.0 * jnp.stack(logabs_list, axis=0)  # (B,)

    # 8) Print the first few
    print("log|ψ| for first 5 walkers:", logprob[:5])

if __name__ == "__main__":
    main()

