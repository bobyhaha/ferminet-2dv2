from ferminet import train
import jax
import jax.numpy as jnp
import kfac_jax
from ferminet.configs import cooper_pair
from ferminet import networks
cfg = cooper_pair.get_config()
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
pos, spins = train.init_electrons(
        subkey,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=256,
        init_width=cfg.mcmc.init_width,
        core_electrons={},
    )
print(pos.shape, spins.shape)