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

"""Unpolarised 14 electron simple cubic homogeneous electron gas."""

from ferminet import base_config
from ferminet.pbc import envelopes_2d
from ferminet.utils import system

import numpy as np


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  volume = (4) * np.pi * (rs**2) * nelec
  length = volume**(1 / 2)
  return length * np.eye(2)


def get_config():
  """Returns config for running unpolarised 14 electron gas with FermiNet."""
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (7, 7)
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0.))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None
  cfg.system.ndim = 2
  lattice = _sc_lattice_vecs(1.0, sum(cfg.system.electrons)) #two dimensional
  kpoints = envelopes_2d.make_kpoints(lattice, cfg.system.electrons)

  cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian_gaussian_2d.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True}
  cfg.network.make_feature_layer_fn = (
      "ferminet.pbc.feature_layer_2d.make_pbc_feature_layer")
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
      "include_r_ae": False
  }
  #Doesn't need distance between X and the electrons
  cfg.network.make_envelope_fn = (
      "ferminet.pbc.envelopes_2d.make_multiwave_envelope")
  cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
  cfg.network.full_det = True
  cfg.optim.reset_if_nan = True
  print("Dimension of system", cfg.system.ndim)
  return cfg
