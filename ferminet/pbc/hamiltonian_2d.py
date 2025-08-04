import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from ferminet import hamiltonian
from ferminet import networks


def make_exact_2d_ewald_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    include_heg_background: bool = True
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
    rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
    area = jnp.abs(jnp.linalg.det(lattice))

    ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
    ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))
    lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
    rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
    rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)

    def real_space_ewald(separation: jnp.ndarray):
        displacements = jnp.linalg.norm(separation - lat_vectors, axis=-1)
        return -jnp.sum(jnp.where(displacements > 1e-12, jnp.log(displacements), 0.0))

    def recp_space_ewald(separation: jnp.ndarray):
        return (2 * jnp.pi / area) * jnp.sum(
            jnp.cos(jnp.dot(rec_vectors, separation)) / rec_vec_square
        )

    def ewald_sum(separation: jnp.ndarray):
        return real_space_ewald(separation) + recp_space_ewald(separation)

    batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

    def atom_electron_potential(ae: jnp.ndarray):
        nelec = ae.shape[0]
        ae = jnp.reshape(ae, [-1, 2])
        ewald = batch_ewald_sum(ae)
        return jnp.sum(-jnp.tile(charges, nelec) * ewald)

    def electron_electron_potential(ee: jnp.ndarray):
        nelec = ee.shape[0]
        ee = jnp.reshape(ee, [-1, 2])
        ewald = batch_ewald_sum(ee)
        ewald = jnp.reshape(ewald, [nelec, nelec])
        ewald = ewald.at[jnp.diag_indices(nelec)].set(0.0)
        return 0.5 * jnp.sum(ewald)

    natom = atoms.shape[0]
    if natom > 1:
        aa = jnp.reshape(atoms, [1, -1, 2]) - jnp.reshape(atoms, [-1, 1, 2])
        aa = jnp.reshape(aa, [-1, 2])
        chargeprods = (charges[..., None] @ charges[..., None].T).flatten()
        ewald = batch_ewald_sum(aa)
        ewald = jnp.reshape(ewald, [natom, natom])
        ewald = ewald.at[jnp.diag_indices(natom)].set(0.0)
        ewald = ewald.flatten()
        atom_atom_potential = 0.5 * jnp.sum(chargeprods * ewald)
    else:
        atom_atom_potential = 0.0

    def potential(ae: jnp.ndarray, ee: jnp.ndarray):
        phase_ae = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ae)
        phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
        phase_prim_ae = phase_ae % 1
        phase_prim_ee = phase_ee % 1
        prim_ae = jnp.einsum('il,jkl->jki', lattice, phase_prim_ae)
        prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
        return jnp.real(
            atom_electron_potential(prim_ae) +
            electron_electron_potential(prim_ee) + atom_atom_potential
        )

    return potential


def local_energy_2d(
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
    if states:
        raise NotImplementedError('Excited states not implemented with PBC.')
    del nspins
    if lattice is None:
        lattice = jnp.eye(2)

    ke = hamiltonian.local_kinetic_energy(f,
                                          use_scan=use_scan,
                                          complex_output=complex_output,
                                          laplacian_method=laplacian_method)

    def _e_l(
        params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        del key
        potential_energy = make_exact_2d_ewald_potential(
            lattice, data.atoms, charges, convergence_radius, heg
        )
        ae, ee, _, _ = networks.construct_input_features(data.positions, data.atoms)
        potential = potential_energy(ae, ee)
        kinetic = ke(params, data)
        return potential + kinetic, None

    return _e_l
