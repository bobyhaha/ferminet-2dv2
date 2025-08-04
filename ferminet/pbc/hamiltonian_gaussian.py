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

"""Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp


def make_ewald_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    include_heg_background: bool = True
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.

  Args:
    lattice: Shape (3, 3). Matrix whose columns are the primitive lattice
      vectors.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    truncation_limit: Integer. Half side length of cube of nearest neighbours
      to primitive cell which are summed over in evaluation of Ewald sum.
      Must be large enough to achieve convergence for the real and reciprocal
      space sums.
    include_heg_background: bool. When True, includes cell-neutralizing
      background term for homogeneous electron gas.

  Returns:
    Callable with signature f(ae, ee), where (ae, ee) are atom-electon and
    electron-electron displacement vectors respectively, which evaluates the
    Coulomb sum for the periodic lattice via the Ewald method.
  """
  #lattice[:, i] is a_i, rec[i] is b_i
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  volume = jnp.abs(jnp.linalg.det(lattice))
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  gamma = (2.8 / volume**(1 / 3))**2
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs) #sorted based on abs value
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=3)))
  #Give the positions of the image charges to be summed over
  # lattice shape (3, 3)
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals) #(n_lattice, 3)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:]) #disregard (0, 0, 0)
  # rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  # lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1) #each element is distance from the origin
  
  def real_space_gaussian(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    #separation is r = r_1 - r_2 and we minus R for the images of r_2. 
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    # jnp.sum(
    #     jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)
    #assume separation takes V =  - alpha * exp(-r_{ij})
    U = 1.0
    sigma = 1/2
    norm = U / ((2 * jnp.pi) ** (3/2) * sigma**3)
    return -norm * jnp.sum(jnp.exp(-0.5 * (displacements ** 2) / sigma ** 2))

  batch_gaussian_sum = jax.vmap(real_space_gaussian, in_axes=(0,)) #over the batch of electrons

  def atom_electron_potential(ae: jnp.ndarray):
    """Evaluates periodic atom-electron potential."""
    nelec = ae.shape[0]
    ae = jnp.reshape(ae, [-1, 3])  # flatten electronxatom axis
    # calculate potential for each ae pair. first dim is the batch
    ewald = batch_gaussian_sum(ae)
    #change the sign to positive? 
    return jnp.sum(-jnp.tile(charges, nelec) * ewald)

  def electron_electron_potential(ee: jnp.ndarray):
    """Evaluates periodic electron-electron potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 3])
    value = batch_gaussian_sum(ee)
    value = jnp.reshape(value, [nelec, nelec])
    value = value.at[jnp.diag_indices(nelec)].set(0.0)
    return 0.5 * jnp.sum(value)

  # Atom-atom potential
  
  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    # Reduce vectors into first unit cell - Ewald summation
    # is only guaranteed to converge close to the origin

    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ee = phase_ee % 1
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    return jnp.real(
        electron_electron_potential(prim_ee))

  return potential


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    heg: bool = True,
    convergence_radius: int = 5,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if states:
    raise NotImplementedError('Excited states not implemented with PBC.')
  del nspins
  if lattice is None:
    lattice = jnp.eye(3)

  ke = hamiltonian.local_kinetic_energy(f,
                                        use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=laplacian_method)

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    potential_energy = make_ewald_potential(
        lattice, data.atoms, charges, convergence_radius, heg
    )
    #ae of shape (nelectrons, natoms, ndim)
    #ee of shape (nelectrons, nelectrons, dim)
    # r_ae of shape (nelectrons, natoms, 1)
    # r_ee of shape (nelectrons, nelectrons, 1)
    ae, ee, _, _ = networks.construct_input_features(
        data.positions, data.atoms)
    potential = potential_energy(ae, ee)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
