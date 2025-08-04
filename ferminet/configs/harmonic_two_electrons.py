# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from ferminet import base_config
from ferminet.pbc import envelopes
from ferminet.utils import system

import numpy as np


def get_config():
  """Returns config for running unpolarised 14 electron gas with FermiNet."""
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (1,1)
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.pretrain.method = None
  cfg.system.molecule = [system.Atom("X", (0., 0.))]
  cfg.system.make_local_energy_fn = "ferminet.hamiltonian_harmonic_2.local_energy"
  cfg.network.full_det = True
  cfg.optim.reset_if_nan = True
  cfg.system.ndim = 2
  cfg.network.network_type = "psiformer"
  return cfg
