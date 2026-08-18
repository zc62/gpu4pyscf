import copy
import ctypes
import math
import cupy
import numpy
import warnings
from scipy.special import erf
from pyscf import gto, lib, scf
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import rks
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.gto.mole import SortedGTO, SortedMole
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import condense
from gpu4pyscf.neo import ks
from gpu4pyscf.qmmm.itrf import _mm_charge_integrals


def general_grad(grad_method):
    if isinstance(grad_method, ComponentGrad):
        return grad_method
    return lib.set_class(ComponentGrad(grad_method),
                         (ComponentGrad, grad_method.__class__))


class ComponentGrad:
    __name_mixin__ = 'Component'

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def get_hcore(self, mol=None, exclude_ecp=False):
        from gpu4pyscf.pbc.gto.int1e import int1e_ipkin
        if mol is None:
            mol = self.mol
        if mol._pseudo:
            raise NotImplementedError('Nuclear gradients for GTH PP')

        if getattr(self.base, 'with_x2c', None):
            raise NotImplementedError('X2C gradients')

        sorted_mol = SortedMole.from_mol(mol, decontract=True)
        h = -int1e_ipkin(sorted_mol) / self.base.mass
        h -= rhf_grad.int1e_ipnuc(mol) * self.base.charge
        if not exclude_ecp and not self.base.is_nucleus and mol.has_ecp():
            h -= get_ecp_ip(mol).sum(axis=0) * self.base.charge
        mm_mol = getattr(getattr(mol, 'super_mol', None), 'mm_mol', None)
        if mm_mol is not None:
            h += _mm_charge_integrals(mm_mol, mol, int1e_grids_ip1) * self.base.charge
        return h

    def _hcore_energy(self, dm0, dme0):
        if not self.base.is_nucleus:
            return super()._hcore_energy(dm0, dme0)

        mol = self.mol
        if mol._pseudo:
            raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

        # Derivatives on the component orbitals.
        from gpu4pyscf.pbc.gto.int1e import int1e_ipkin
        sorted_mol = SortedMole.from_mol(mol, decontract=True)
        h1 = -int1e_ipkin(sorted_mol) / self.base.mass
        h1 -= rhf_grad.int1e_ipnuc(mol) * self.base.charge
        dh = rhf_grad.contract_h1e_dm(mol, h1, dm0, hermi=1)
        s1 = cupy.asarray(self.get_ovlp(mol))
        dh -= rhf_grad.contract_h1e_dm(mol, s1, dme0, hermi=1)

        # Derivatives on the classical potential centers.
        charges = cupy.asarray(mol.atom_charges() * self.base.charge)
        if cupy.any(charges != 0):
            dh += int1e_grids_ip2(mol, mol.atom_coords(),
                                   charges=charges, dm=dm0).T.get()
        return dh

    def energy_ee(self, mol=None, dm=None, verbose=None):
        if self.base.is_nucleus:
            if mol is None:
                mol = self.mol
            return numpy.zeros((mol.natm, 3))
        return super().energy_ee(mol, dm)

    def get_veff(self, mol=None, dm=None):
        if self.base.is_nucleus:
            if mol is None:
                mol = self.mol
            return cupy.zeros((3, mol.nao, mol.nao))
        assert abs(self.base.charge) == 1
        return super().get_veff(mol, dm)

    def extra_force(self, atom_id=None):
        assert atom_id is None
        mm_mol = getattr(getattr(self.mol, 'super_mol', None), 'mm_mol', None)
        if mm_mol is None:
            return super().extra_force()

        h1 = _mm_charge_integrals(mm_mol, self.mol, int1e_grids_ip1) * self.base.charge
        dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        e1_grad = rhf_grad.contract_h1e_dm(self.mol, h1, dm, hermi=1)
        e1_grad += super().extra_force()
        return e1_grad

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise AttributeError


rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in.restype = ctypes.c_int
rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in_no_self.restype = ctypes.c_int


def _j_intercomponent_energy_per_atom(vhfopt, mols, dms, group1_size,
                                      verbose=None):
    '''
    Computes the first-order derivatives of the intercomponent Coulomb energy
    per atom.
    '''
    # Evaluate both directions of the intercomponent Coulomb gradient.
    # Component masks select local density pairs and cross-component integral
    # pairs in the combined molecule.
    log = logger.new_logger(vhfopt.mol, verbose)
    cput0 = log.init_timer()
    mol = vhfopt.sorted_mol

    component_mols = [SortedGTO.from_mol(mol_t, decontract=True,
                                         diffuse_cutoff=0.3)
                      for mol_t in mols]
    dms = [mol_t.apply_C_mat_CT(cupy.asarray(dm_t, order='C'))
           for mol_t, dm_t in zip(component_mols, dms)]

    orig_shell_component = numpy.hstack([
        numpy.full(mol_t.nbas, i, dtype=numpy.int32)
        for i, mol_t in enumerate(component_mols)])
    orig_shell_local = numpy.hstack([
        numpy.arange(mol_t.nbas, dtype=numpy.int32)
        for mol_t in component_mols])
    shell_component = orig_shell_component[mol.sorted_idx]
    shell_local = orig_shell_local[mol.sorted_idx]
    local_ao_loc = numpy.empty(mol.nbas, dtype=numpy.int32)
    component_nao = numpy.asarray([mol_t.nao for mol_t in component_mols],
                                  dtype=numpy.int32)
    local_shell = []
    for i, mol_t in enumerate(component_mols):
        inv_sorted = numpy.empty_like(mol_t.sorted_idx)
        inv_sorted[mol_t.sorted_idx] = numpy.arange(mol_t.nbas)
        idx = numpy.where(shell_component == i)[0]
        shell_t = inv_sorted[shell_local[idx]]
        local_shell.append(shell_t)
        local_ao_loc[idx] = mol_t.ao_loc[shell_t]
    if group1_size is not None:
        shell_group = shell_component >= group1_size

    uniq_l_ctr = mol.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = numpy.append(0, numpy.cumsum(mol.l_ctr_counts))
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= rhf_grad.LMAX

    log_cutoff = math.log(vhfopt.direct_scf_tol)
    dm_penalty = 0
    diffuse_exps, diffuse_ctr_coef = rhf_grad.extract_pgto_params(mol, 'diffuse')
    n_groups = len(uniq_l_ctr)
    tasks = ((i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1))

    def proc():
        device_id = cupy.cuda.device.get_device_id()
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        timing_collection = rhf_grad._TimingCollector(log.timer_debug1)
        kern_counts = 0
        if group1_size is None:
            kern = rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in_no_self
        else:
            kern = rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in

        _dms = [cupy.asarray(dm_t, order='C') for dm_t in dms]
        dm_ptrs = []
        for dm_t in _dms:
            dm_ptrs.append(dm_t.data.ptr)
        dm_ptrs = cupy.asarray(numpy.array(dm_ptrs))
        ejk = cupy.zeros((mol.natm, 3))
        dm_cond = cupy.full((mol.nbas, mol.nbas), -1e30, dtype=numpy.float32)
        for i, (dm_t, mol_t) in enumerate(zip(_dms, component_mols)):
            dm_cond_t = cupy.log(condense('absmax', dm_t[None], mol_t.ao_loc) + 1e-300).astype(numpy.float32)
            idx = numpy.where(shell_component == i)[0]
            dm_cond[numpy.ix_(idx, idx)] = dm_cond_t[numpy.ix_(local_shell[i], local_shell[i])]
        _diffuse_exps = cupy.asarray(diffuse_exps, dtype=numpy.float32)
        _shell_component = cupy.asarray(shell_component, dtype=numpy.int32)
        if group1_size is not None:
            _shell_group = cupy.asarray(shell_group)
        _local_ao_loc = cupy.asarray(local_ao_loc, dtype=numpy.int32)
        _component_nao = cupy.asarray(component_nao, dtype=numpy.int32)
        bas_pair_cache = {k: [cupy.asarray(x) for x in v]
                          for k, v in vhfopt.bas_pair_cache.items()}
        rys_envs = vhfopt.rys_envs
        workers = rhf_grad.gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs
        pool = cupy.empty(workers*rhf_grad.QUEUE_DEPTH+1, dtype=numpy.int32)
        dd_pool = cupy.empty((workers, rhf_grad.DD_CACHE_MAX), dtype=numpy.float64)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in tasks:
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            pair_ij_mapping0, q_cond_ij0, s_cond_ij0 = bas_pair_cache[i,j]
            pair_kl_mapping0, q_cond_kl0, s_cond_kl0 = bas_pair_cache[k,l]
            if pair_ij_mapping0.size == 0 or pair_kl_mapping0.size == 0:
                continue
            ish_ij = pair_ij_mapping0 // mol.nbas
            jsh_ij = pair_ij_mapping0 % mol.nbas
            same_component_ij = _shell_component[ish_ij] == _shell_component[jsh_ij]
            ish_kl = pair_kl_mapping0 // mol.nbas
            jsh_kl = pair_kl_mapping0 % mol.nbas
            same_component_kl = _shell_component[ish_kl] == _shell_component[jsh_kl]
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            scheme = rhf_grad._ejk_quartets_scheme(
                mol, uniq_l_ctr[[i, j, k, l]])
            component_groups = ((0, 1), (1, 0)) if group1_size is not None else ((None, None),)
            for comp_ij, comp_kl in component_groups:
                if comp_ij is None:
                    pair_mask = same_component_ij
                else:
                    pair_mask = same_component_ij & (_shell_group[ish_ij] == comp_ij)
                pair_ij_mapping = pair_ij_mapping0[pair_mask]
                q_cond_ij = q_cond_ij0[pair_mask]
                s_cond_ij = s_cond_ij0[pair_mask]

                if comp_kl is None:
                    pair_mask = same_component_kl
                else:
                    pair_mask = same_component_kl & (_shell_group[ish_kl] == comp_kl)
                pair_kl_mapping = pair_kl_mapping0[pair_mask]
                q_cond_kl = q_cond_kl0[pair_mask]
                s_cond_kl = s_cond_kl0[pair_mask]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_shell_component.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_local_ao_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_component_nao.data.ptr, ctypes.c_void_p),
                    rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
        return ejk, kern_counts, timing_collection

    results = rhf_grad.multi_gpu.run(proc, non_blocking=True)
    ejk = rhf_grad.multi_gpu.array_reduce([x[0] for x in results], inplace=True)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', sum(x[1] for x in results))
        rhf_grad._TimingCollector.summary(log.debug1, (x[2] for x in results))

    log.timer_debug1('grad j energy', *cput0)
    return ejk.get()


def _grad_eri_group(mf_grad, dms, keys1, keys2, atmlst):
    '''Evaluate all Coulomb-gradient edges between two component groups.'''
    mf = mf_grad.base
    mol1 = mf.components[keys1[0]].mol
    for t in keys1[1:]:
        mol1 = mol1 + mf.components[t].mol
    if keys2 is None:
        vhfopt = rhf_grad._VHFOpt(mol1, mf.direct_scf_tol).build()
        keys = keys1
        group1_size = None
    else:
        mol2 = mf.components[keys2[0]].mol
        for t in keys2[1:]:
            mol2 = mol2 + mf.components[t].mol
        vhfopt = rhf_grad._VHFOpt(mol1 + mol2, mf.direct_scf_tol).build()
        keys = keys1 + keys2
        group1_size = len(keys1)
    de = _j_intercomponent_energy_per_atom(
        vhfopt, [mf.components[t].mol for t in keys],
        [mf.components[t].charge*dms[t] for t in keys], group1_size)
    de = de.reshape(len(keys), mf.mol.natm, 3).sum(axis=0)
    return de[list(atmlst)]


def grad_eri(mf_grad, dm0, interactions, atmlst):
    mf = mf_grad.base
    interactions = list(interactions)
    component_keys = [t for t in mf.components
                      if any(t in pair for pair, _ in interactions)]
    unrestricted = {}
    for (t1, t2), interaction in interactions:
        unrestricted[t1] = interaction.mf1_unrestricted
        unrestricted[t2] = interaction.mf2_unrestricted
    dms = {}
    for t in component_keys:
        dm = cupy.asarray(dm0[t])
        if unrestricted[t]:
            assert dm.ndim > 2 and dm.shape[0] == 2
            dm = dm[0] + dm[1]
        dms[t] = dm

    de = numpy.zeros((len(atmlst), 3))
    # Component IDs remove nuclear self-interactions from the grouped N-N job.
    nuclear_keys = [t for t in component_keys if t != 'e']
    if 'e' in component_keys and nuclear_keys:
        de += _grad_eri_group(mf_grad, dms, ['e'], nuclear_keys,
                              atmlst)
    component_keys = nuclear_keys
    if len(component_keys) > 1:
        de += _grad_eri_group(mf_grad, dms, component_keys, None, atmlst)
    return de


def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
             atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if atmlst is None:
        atmlst = range(mol.natm)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    de = grad_eri(mf_grad, dm0, mf.interactions.items(), atmlst)

    if mf_grad.verbose >= logger.DEBUG:
        log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
        log.debug('gradients of Coulomb interaction')
        rhf_grad_cpu._write(log, mol, de, atmlst)
    return de


def grad_epc(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    de = numpy.zeros((len(atmlst), 3))
    if not hasattr(mf, 'epc') or mf.epc is None:
        return de

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    mf_e = mf.components['e']
    mol_e = mf_e.mol
    dm_e = cupy.asarray(dm0['e'])
    if isinstance(mf_e, scf.uhf.UHF):
        assert dm_e.ndim > 2 and dm_e.shape[0] == 2
        dm_e = dm_e[0] + dm_e[1]

    grids_e = mf_e.grids
    if grids_e.coords is None:
        rks.initialize_grids(mf_e, mol_e, dm_e)
    elec_grids_hash = ks._hash_grids(grids_e)
    grids_changed = (mf._elec_grids_hash != elec_grids_hash)
    if grids_changed and mf._epc_n_types is not None:
        if len(mf._epc_n_types) > 0:
            mf._skip_epc = False
    if getattr(mf, '_skip_epc', False):
        return de

    if mf._epc_n_types is None:
        n_types = []
        for t_pair, interaction in mf.interactions.items():
            if interaction._need_epc():
                if t_pair[0].startswith('n'):
                    n_type  = t_pair[0]
                else:
                    n_type  = t_pair[1]
                n_types.append(n_type)
        mf._epc_n_types = n_types
    else:
        n_types = mf._epc_n_types
    if len(n_types) == 0:
        mf._skip_epc = True
        return de

    mol_n_all = mf.components[n_types[0]].mol
    n_slices = {}
    p0 = 0
    for n_type in n_types:
        mol_n_t = mf.components[n_type].mol
        if n_type != n_types[0]:
            mol_n_all = gto.conc_mol(mol_n_all, mol_n_t)
        p1 = p0 + mol_n_t.nao
        n_slices[n_type] = (p0, p1)
        p0 = p1

    ni_n = numint.NumInt()
    ni_n.build(mol_n_all, grids_e.coords)
    if mf.grids is None or grids_changed:
        grids = copy.copy(grids_e)
        grids.mol = mol_n_all
        grids._non0ao_idx = None
        non0ao_idx_n = grids.get_non0ao_idx(ni_n.gdftopt)
        block_ids = [i for i, x in enumerate(non0ao_idx_n) if len(x[1]) > 0]
        if len(block_ids) == 0:
            mf._skip_epc = True
            return de
        starts = (cupy.asarray(block_ids)[:,None] * numint.MIN_BLK_SIZE
                  + cupy.arange(numint.MIN_BLK_SIZE))
        valid_idx = starts[starts < grids_e.coords.shape[0]]
        mf.grids = copy.copy(grids_e)
        mf.grids.coords = grids_e.coords[valid_idx]
        mf.grids.weights = grids_e.weights[valid_idx]
        if getattr(grids_e, 'atm_idx', None) is not None:
            mf.grids.atm_idx = grids_e.atm_idx[valid_idx]
        if getattr(grids_e, 'quadrature_weights', None) is not None:
            mf.grids.quadrature_weights = grids_e.quadrature_weights[valid_idx]
        mf.grids.non0tab = None
        mf.grids.screen_index = None
        mf.grids._non0ao_idx = None
        mf._elec_grids_hash = elec_grids_hash
    grids = mf.grids

    ni_e = mf_e._numint
    if ni_e.gdftopt is None:
        ni_e.build(mol_e, grids.coords)
    opt_e = ni_e.gdftopt
    sorted_mol_e = opt_e._sorted_mol
    dm_e_sorted = opt_e.sort_orbitals(dm_e, axis=[0,1])

    opt_n = ni_n.gdftopt
    sorted_mol_n = opt_n._sorted_mol
    ao_idx_n = cupy.asarray(opt_n._ao_idx)
    grids_n = copy.copy(grids)
    grids_n.mol = mol_n_all
    grids_n._non0ao_idx = None
    non0ao_idx_n = grids_n.get_non0ao_idx(opt_n)

    dm_n_all = cupy.zeros((mol_n_all.nao, mol_n_all.nao))
    for n_type in n_types:
        n0, n1 = n_slices[n_type]
        dm_n_all[n0:n1,n0:n1] = cupy.asarray(dm0[n_type])
    dm_n_all = opt_n.sort_orbitals(dm_n_all, axis=[0,1])

    nao_e = sorted_mol_e.nao
    nao_n = sorted_mol_n.nao
    vxc_e = cupy.zeros((3, nao_e, nao_e))
    vxc_n = cupy.zeros((3, nao_n, nao_n))

    p1 = 0
    for ao_e, idx_e, weight, coords in ni_e.block_loop(
            sorted_mol_e, grids, nao_e, deriv=1, strict_grid_order=True):
        p0, p1 = p1, p1 + weight.size
        if len(idx_e) == 0:
            continue
        dm_e_mask = dm_e_sorted[idx_e[:,None],idx_e]
        rho_e = numint.eval_rho(sorted_mol_e, ao_e[0], dm_e_mask, hermi=1)
        rho_e = cupy.maximum(rho_e, 0)
        common = ks.precompute_epc_electron(mf.epc, rho_e)

        vxc_e_grid = 0
        block_id = p0 // numint.MIN_BLK_SIZE
        pad, idx_n, non0shl_idx, ctr_offsets_slice, ao_loc_slice = \
            non0ao_idx_n[block_id]
        if len(idx_n) == 0:
            continue

        ao_n = numint.eval_ao(
            sorted_mol_n, coords, deriv=1, nao_slice=len(idx_n),
            shls_slice=non0shl_idx, ao_loc_slice=ao_loc_slice,
            ctr_offsets_slice=ctr_offsets_slice, gdftopt=opt_n,
            transpose=False)
        if pad > 0:
            ao_n[:,-pad:] = 0.0
        orig_idx_n = ao_idx_n[idx_n]

        # TODO: Group the component-local nuclear density and derivative-matrix
        # contractions without forming a padded block-diagonal DM.
        for n_type in n_types:
            n0, n1 = n_slices[n_type]
            mask_n = (orig_idx_n >= n0) & (orig_idx_n < n1)
            if not cupy.any(mask_n):
                continue
            idx_n_t = idx_n[mask_n]
            ao_n_t = ao_n[:,mask_n]
            dm_n_mask = dm_n_all[idx_n_t[:,None],idx_n_t]
            rho_n = numint.eval_rho(sorted_mol_n, ao_n_t[0],
                                    dm_n_mask, hermi=1)
            rho_n = cupy.maximum(rho_n, 0)
            _, vxc_n_grid, vxc_e_grid_t = ks.eval_epc(common, rho_n)
            vxc_e_grid += vxc_e_grid_t

            aow_n = numint._scale_ao(ao_n_t[0], weight * vxc_n_grid)
            vtmp = rks_grad._d1_dot_(ao_n_t[1:4], aow_n.T)
            vxc_n[:,idx_n_t[:,None],idx_n_t] += vtmp

        aow_e = numint._scale_ao(ao_e[0], weight * vxc_e_grid)
        vtmp = rks_grad._d1_dot_(ao_e[1:4], aow_e.T)
        vxc_e[:,idx_e[:,None],idx_e] += vtmp

    vxc_e = opt_e.unsort_orbitals(vxc_e, axis=[1,2])
    vxc_n = opt_n.unsort_orbitals(vxc_n, axis=[1,2])

    aoslices = mol_e.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        de[i0] -= cupy.einsum('xij,ij->x', vxc_e[:,p0:p1],
                              dm_e[p0:p1]).real.get() * 2

    for n_type in n_types:
        n0, n1 = n_slices[n_type]
        dm_n = cupy.asarray(dm0[n_type])
        vxc_n_t = vxc_n[:,n0:n1,n0:n1]
        aoslices = mf.components[n_type].mol.aoslice_by_atom()
        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            if p1 > p0:
                de[i0] -= cupy.einsum('xij,ij->x', vxc_n_t[:,p0:p1],
                                      dm_n[p0:p1]).real.get() * 2
    return de


class Gradients(rhf_grad.GradientsBase):
    def __init__(self, mf):
        super().__init__(mf)
        self.grid_response = None
        self.components = {}
        for t, comp in self.base.components.items():
            self.components[t] = general_grad(comp.nuc_grad_method())
        self._keys = self._keys.union(['grid_response', 'components'])

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(self.mol)
        if sorted(self.components.keys()) == sorted(self.mol.components.keys()):
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
        else:
            self.components.clear()
            for t, comp in self.base.components.items():
                self.components[t] = general_grad(comp.nuc_grad_method())
        return self

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None:
            mol = self.mol
        g_qm = self.components['e'].grad_nuc(mol.components['e'], atmlst)
        if mol.mm_mol is None:
            return g_qm

        mm_mol = mol.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            radii = 1 / numpy.sqrt(expnts)

        mol_e = mol.components['e']
        g_mm = numpy.empty((mol_e.natm, 3))
        for i in range(mol_e.natm):
            q1 = mol_e.atom_charge(i)
            r1 = mol_e.atom_coord(i)
            r = lib.norm(r1 - coords, axis=1)
            if mm_mol.charge_model != 'gaussian':
                coulkern = 1 / r**3
            else:
                coulkern = erf(r/radii)/r - 2/(numpy.sqrt(numpy.pi)*radii) \
                         * numpy.exp(-expnts*r**2)
                coulkern = coulkern / r**2
            g_mm[i] = -q1 * numpy.einsum('i,ix,i->x',
                                         charges, r1 - coords, coulkern)
        if atmlst is not None:
            g_mm = g_mm[atmlst]
        return g_qm + g_mm

    def symmetrize(self, de, atmlst=None):
        return rhf_grad_cpu.symmetrize(self.mol.components['e'], de, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff
        if mo_occ is None:
            mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if self.grid_response and hasattr(self.base, 'epc') and self.base.epc is not None:
            raise NotImplementedError('Grid response for NEO EPC gradients')

        # Component gradient implementations return all atoms; apply atmlst
        # only after assembling the full multicomponent gradient.
        de = 0
        # TODO: Batch nuclear one-electron gradient integrals without forming
        # cross-component AO blocks.  Each grad_elec call currently launches
        # the small hcore, overlap, and optional MM derivative kernels separately.
        for t, comp in self.components.items():
            if self.grid_response is not None and hasattr(comp, 'grid_response'):
                comp.grid_response = self.grid_response
            de += comp.grad_elec(mo_energy=mo_energy[t],
                                 mo_coeff=mo_coeff[t],
                                 mo_occ=mo_occ[t])

        de += self.grad_int(mo_energy, mo_coeff, mo_occ)
        if hasattr(self.base, 'epc') and self.base.epc is not None:
            de += self.grad_epc(mo_energy, mo_coeff, mo_occ)

        self.de = de + self.grad_nuc()
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)
        if self.base.do_disp():
            self.de += self.components['e'].get_dispersion()
        if atmlst is not None:
            self.de = self.de[atmlst]
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    grad_int = grad_int
    grad_epc = grad_epc
    as_scanner = rhf_grad.as_scanner
    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def grad_hcore_mm(self, dm=None, mol=None):
        if mol is None:
            mol = self.mol
        mm_mol = mol.mm_mol
        if mm_mol is None:
            warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
            return None
        if dm is None:
            dm = self.base.make_rdm1()

        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        expnts = mm_mol.get_zetas()
        g = numpy.zeros_like(coords)
        # TODO: Contract the MM potential derivatives for all component-local
        # DMs in one launch without forming a padded block-diagonal DM.
        for t, comp in self.base.components.items():
            dm_comp = cupy.asarray(dm[t])
            if dm_comp.ndim > 2:
                dm_comp = dm_comp[0] + dm_comp[1]
            g += int1e_grids_ip2(comp.mol, coords, dm=dm_comp,
                                 charge_exponents=expnts).T.get() \
                    * charges[:,None] * comp.charge
        return g

    contract_hcore_mm = grad_hcore_mm

    def grad_nuc_mm(self, mol=None):
        if mol is None:
            mol = self.mol
        mm_mol = mol.mm_mol
        if mm_mol is None:
            warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
            return None
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            radii = 1 / numpy.sqrt(expnts)
        g_mm = numpy.zeros_like(coords)
        mol_e = mol.components['e']
        for i in range(mol_e.natm):
            q1 = mol_e.atom_charge(i)
            r1 = mol_e.atom_coord(i)
            r = lib.norm(r1 - coords, axis=1)
            if mm_mol.charge_model != 'gaussian':
                coulkern = 1 / r**3
            else:
                coulkern = erf(r/radii)/r - 2/(numpy.sqrt(numpy.pi)*radii) \
                         * numpy.exp(-expnts*r**2)
                coulkern = coulkern / r**2
            g_mm += q1 * numpy.einsum('i,ix,i->ix',
                                      charges, r1 - coords, coulkern)
        return g_mm

    def grad_mm(self, dm=None, mol=None):
        return self.grad_hcore_mm(dm, mol) + self.grad_nuc_mm(mol)


Grad = Gradients
