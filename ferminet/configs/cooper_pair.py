from ferminet import base_config
from ferminet.utils import system
from ferminet import train
from absl import logging
import sys
# Settings in a config files are loaded by executing the the get_config
# function.
def get_config():
  # Get default options.
  cfg = base_config.default()
  # Set up molecule
  cfg.system.electrons = (1,1)
  cfg.system.molecule = [system.Atom("X", (0., 0.))]
  cfg.system.make_local_energy_fn =  "ferminet.hamiltonian_gaussian.local_energy"
  # Set training hyperparameters
  cfg.system.ndim = 2
  cfg.batch_size = 256
  cfg.pretrain.iterations = 0
  cfg.log.save_frequency = 1
  cfg.network.network_type = "psiformer"
  return cfg

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

cfg = get_config()
train.train(cfg)
