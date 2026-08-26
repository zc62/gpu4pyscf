import cupy
import numpy
from pyscf import scf as scf_cpu
from pyscf.neo import cdft as cdft_cpu
from pyscf.neo import hf as hf_cpu
from pyscf.neo import ks as ks_cpu
from gpu4pyscf import scf
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.neo import hf, ks


def _constraint_groups(mf, keys, fock0, s1e):
    groups = {}
    key_set = set(keys)
    # Preserve the position batches while matching eigensolver input dtypes.
    for batch_keys, position_batch in mf._int1e_r_batches:
        for i, t in enumerate(batch_keys):
            if t in key_set:
                key = (fock0[t].shape, fock0[t].dtype, s1e[t].dtype,
                       position_batch.dtype)
                groups.setdefault(key, []).append((t, position_batch, i))

    constraint_groups = []
    key_index = {t: i for i, t in enumerate(keys)}
    for group in groups.values():
        group_keys = [item[0] for item in group]
        components = [mf.components[t] for t in group_keys]
        position_batch = group[0][1]
        position_indices = [item[2] for item in group]
        if len(group_keys) != len(position_batch):
            position_batch = position_batch[position_indices]
        constraint_groups.append((
            cupy.asarray([key_index[t] for t in group_keys]),
            cupy.stack([fock0[t] for t in group_keys]),
            cupy.stack([s1e[t] for t in group_keys]),
            position_batch,
            cupy.arange(len(group_keys)),
            cupy.asarray([comp.nuc_occ_state for comp in components])))
    return constraint_groups


def _evaluate_position_response(f_lagrange, constraint_groups,
                                gap_tol=1e-14,
                                with_jacobian=True):
    deviations = cupy.empty_like(f_lagrange)
    if with_jacobian:
        jacobians = cupy.empty((len(f_lagrange), 3, 3))
        degenerate = cupy.zeros(len(f_lagrange), dtype=bool)
    else:
        jacobians = degenerate = None
    for indices, fock0, overlap, int1e_r, rows, states in constraint_groups:
        # Equal-sized nuclear components share one generalized eigensolve.
        multipliers = f_lagrange[indices]
        fock = fock0 + cupy.einsum('txij,tx->tij', int1e_r, multipliers)
        mo_energy, mo_coeff = hf._eig_batch(fock, overlap)
        occupied = mo_coeff[rows,:,states]
        deviation = cupy.einsum('tp,txpq,tq->tx', occupied.conj(),
                                 int1e_r, occupied).real
        deviations[indices] = deviation
        if with_jacobian:
            # Frozen-Fock occupied-orbital response gives d<r>/df.
            coupling = cupy.einsum('tp,txpq,tqa->txa', occupied.conj(),
                                   int1e_r, mo_coeff)
            energy_gap = mo_energy - mo_energy[rows,states,None]
            mask = cupy.abs(energy_gap) > gap_tol
            mask[rows,states] = False
            inverse_gap = cupy.zeros_like(energy_gap)
            inverse_gap[mask] = 1 / energy_gap[mask]
            jacobian = -2 * cupy.einsum('txa,tya,ta->txy', coupling,
                                        coupling.conj(), inverse_gap).real
            jacobian = (jacobian + jacobian.swapaxes(-1, -2)) * .5
            is_degenerate = cupy.count_nonzero(mask, axis=1) != \
                mo_energy.shape[1] - 1
            jacobians[indices] = jacobian
            degenerate[indices] = is_degenerate
    return deviations, jacobians, degenerate


def get_position_error(mf, fock, s1e):
    '''Return concatenated position-constraint errors for quantum nuclei.'''
    keys = sorted(t for t in mf.components if t.startswith('n'))
    f_lagrange_array = cupy.zeros((len(keys), 3))
    constraint_groups = _constraint_groups(mf, keys, fock, s1e)
    deviations, _, _ = _evaluate_position_response(
        f_lagrange_array, constraint_groups, with_jacobian=False)
    return deviations.ravel()


def update_lagrange_multipliers(mf, fock0, s1e, one_step=False,
                                max_cycle=50, tol=1e-15):
    '''Update the CNEO Lagrange multipliers with batched Newton steps.'''
    gap_tol = 1e-14
    minimum_step = 0.01
    keys = sorted(t for t in mf.components if t.startswith('n'))
    atom_indices = [mf.components[t].mol.atom_index for t in keys]
    f_lagrange = mf.f[atom_indices]
    constraint_groups = _constraint_groups(mf, keys, fock0, s1e)
    ncycle = 1 if one_step else max_cycle

    for cycle in range(ncycle):
        deviations, jacobians, degenerate = _evaluate_position_response(
            f_lagrange, constraint_groups, gap_tol)
        if not one_step and cupy.max(cupy.abs(deviations)) < tol:
            logger.debug(mf, 'CNEO constraint Newton update converged at cycle %d', cycle)
            break
        if cupy.any(degenerate):
            logger.warn(mf, 'Falling back to numerical position Jacobian for '
                        'degenerate nuclear orbitals')
            displacements = cupy.eye(3) * 1e-3
            numerical_jacobian = []
            for displacement in displacements:
                displaced = f_lagrange + displacement
                displaced_deviations, _, _ = _evaluate_position_response(
                    displaced, constraint_groups, with_jacobian=False)
                numerical_jacobian.append(
                    (displaced_deviations - deviations) / 1e-3)
            numerical_jacobian = cupy.stack(numerical_jacobian, axis=2)
            jacobians = cupy.where(degenerate[:,None,None],
                                   numerical_jacobian, jacobians)

        try:
            direction = cupy.linalg.solve(jacobians, -deviations)
        except cupy.linalg.LinAlgError:
            direction = cupy.einsum('txy,ty->tx', cupy.linalg.pinv(jacobians),
                                    -deviations)
        deviation_norm = cupy.linalg.norm(deviations, axis=1)
        direction_norm = cupy.linalg.norm(direction, axis=1)
        unconverged = cupy.max(cupy.abs(deviations), axis=1) >= tol
        valid = unconverged & (direction_norm != 0) & cupy.isfinite(direction_norm)
        step_size = cupy.ones(len(keys))
        trial = f_lagrange + direction
        trial_deviations, _, _ = _evaluate_position_response(
            trial, constraint_groups, gap_tol)
        trial_norm = cupy.linalg.norm(trial_deviations, axis=1)

        while cupy.any(valid & (trial_norm >= deviation_norm) &
                       (step_size >= minimum_step)):
            rejected = valid & (trial_norm >= deviation_norm) & \
                (step_size >= minimum_step)
            slope = -deviation_norm / (step_size * direction_norm)
            denominator = 2 * (trial_norm - deviation_norm - slope)
            quadratic_step = cupy.maximum(-slope / denominator, 0.1)
            halve = (denominator == 0) | ~cupy.isfinite(denominator)
            factor = cupy.where(halve, 0.5, quadratic_step)
            step_size = cupy.where(rejected, step_size * factor, step_size)
            trial = f_lagrange + step_size[:,None] * direction
            trial_deviations, _, _ = _evaluate_position_response(
                trial, constraint_groups, gap_tol)
            trial_norm = cupy.linalg.norm(trial_deviations, axis=1)

        accepted = valid & (trial_norm < deviation_norm)
        f_lagrange = cupy.where(accepted[:,None], trial, f_lagrange)
        deviations = cupy.where(accepted[:,None], trial_deviations, deviations)
        if not one_step and cupy.max(cupy.abs(deviations)) < tol:
            logger.debug(mf, 'CNEO constraint Newton update converged at cycle %d', cycle+1)
            break
    else:
        if not one_step:
            logger.warn(mf, 'CNEO constraint Newton update did not converge in %d cycles',
                        max_cycle)

    mf.f[atom_indices] = f_lagrange
    for i, t in enumerate(keys):
        ia = mf.components[t].mol.atom_index
        logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                     (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
        logger.debug(mf, 'Position deviation: %s', deviations[i])
    return deviations.ravel()


class CDFT(ks.KS):
    _keys = ks.KS._keys.union({'f'})

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.f = cupy.zeros((mol.natm, 3))
        self._setup_position_matrices()

    def _setup_position_matrices(self):
        '''Set up position matrices for each quantum nucleus for constraint'''
        position_matrices = {}
        for t, comp in self.components.items():
            if t.startswith('n'):
                if comp.mol.symmetry:
                    raise NotImplementedError('Symmetry adapted CDFT position '
                                              'constraint is not implemented')
                comp.nuclear_expect_position = \
                    comp.mol.atom_coord(comp.mol.atom_index)
                position_matrices[t] = comp.mol.intor_symmetric(
                    'int1e_r', comp=3)
        keys = [t for t in self.components if t.startswith('n')]
        s1e = self.get_ovlp()
        groups = {}
        for t in keys:
            int1e_r = position_matrices[t]
            key = (int1e_r.shape, int1e_r.dtype, s1e[t].dtype)
            groups.setdefault(key, []).append(t)
        self._int1e_r_batches = []
        for group in groups.values():
            matrices = cupy.asarray(numpy.stack([
                position_matrices[t] for t in group]))
            origins = cupy.asarray(numpy.stack([
                self.components[t].nuclear_expect_position for t in group]))
            overlap = cupy.stack([s1e[t] for t in group])
            # Position matrix with origin shifted to nuclear expectation position
            matrices -= origins[:,:,None,None] * overlap[:,None]
            self._int1e_r_batches.append((group, matrices))
            for i, t in enumerate(group):
                self.components[t].int1e_r = matrices[i]
                self.components[t].int1e_r_symm = None

    def get_fock_add_cdft(self):
        '''Get additional Fock terms from constraints'''
        f_add = {}
        for keys, int1e_r in self._int1e_r_batches:
            atom_indices = [self.components[t].mol.atom_index for t in keys]
            multipliers = self.f[atom_indices]
            f_add_batch = cupy.einsum('txij,tx->tij', int1e_r, multipliers)
            for i, t in enumerate(keys):
                f_add[t] = f_add_batch[i]
        return f_add

    dip_moment = cdft_cpu.CDFT.dip_moment

    def reset(self, mol=None):
        super().reset(mol=mol)
        self.f = cupy.zeros((self.mol.natm, 3))
        self._setup_position_matrices()
        return self

    def to_cpu(self):
        obj = cdft_cpu.CDFT(self.mol, unrestricted=self.unrestricted,
                            xc=self.xc_e, epc=self.epc)
        for key in self._keys:
            if key in ('components', 'interactions'):
                continue
            if hasattr(self, key):
                setattr(obj, key, hf._to_cpu(getattr(self, key)))
        obj.components = {t: comp.to_cpu() for t, comp in self.components.items()}
        obj._setup_position_matrices()
        obj.interactions = hf_cpu.generate_interactions(
            obj.components, ks_cpu.InteractionCorrelation,
            obj.max_memory, obj.direct_scf_tol, epc=obj.epc)
        if isinstance(obj.components['e'], scf_cpu.hf.KohnShamDFT):
            obj._numint = obj.components['e']._numint
        else:
            obj._numint = None
        obj.grids = None
        obj._elec_grids_hash = None
        obj._epc_n_types = None
        obj._skip_epc = False
        return obj

    to_gpu = utils.to_gpu


def from_cpu(mf):
    out = CDFT(mf.mol, unrestricted=mf.unrestricted, xc=mf.xc_e, epc=mf.epc)
    for key, val in mf.__dict__.items():
        if key in ('components', 'interactions', 'grids', '_elec_grids_hash',
                   '_epc_n_types', '_skip_epc', '_numint'):
            continue
        setattr(out, key, hf._to_gpu(val))
    out.components = {t: comp.to_gpu() for t, comp in mf.components.items()}
    out._setup_position_matrices()
    if isinstance(out.components['e'], scf.hf.KohnShamDFT):
        out._numint = out.components['e']._numint
    else:
        out._numint = None
    out.grids = None
    out._elec_grids_hash = None
    out._epc_n_types = None
    out._skip_epc = False
    out.interactions = ks.hf_cpu.generate_interactions(
        out.components, ks.InteractionCorrelation,
        out.max_memory, out.direct_scf_tol, epc=out.epc)
    return out
