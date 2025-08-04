from ferminet import base_config
from ferminet.utils import system

# Settings in a config files are loaded by executing the the get_config
# function.
def get_config():
  # Get default options.
  cfg = base_config.default()
  # Set up molecule
  cfg.system.electrons = (1,1)
  cfg.system.molecule = [system.Atom('H', (0, -1)), system.Atom('H', (0, 1))]

  # Set training hyperparameters
  cfg.system.ndim = 2
  cfg.batch_size = 256
  cfg.pretrain.iterations = 0
  cfg.log.save_frequency = 0.1

  return cfg

