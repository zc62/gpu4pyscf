import ctypes
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.gto.mole import ANG_OF, PTR_EXP, conc_env

from gpu4pyscf.df import j_engine_3c2e as df_j_engine_3c2e
from gpu4pyscf.df.int3c2e_bdiv import _conc_locs, LMAX, L_AUX_MAX, THREADS
from gpu4pyscf.gto.mole import SortedGTO, PTR_BAS_COORD, RysIntEnvVars
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, transpose_sum
from gpu4pyscf.scf.j_engine import libvhf_md
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE

libvhf_md.dm_to_Rt_multi_in.restype = ctypes.c_int
libvhf_md.contract_int3c2e_dm_multi_out.restype = ctypes.c_int
libvhf_md.contract_int3c2e_auxvec_multi_in.restype = ctypes.c_int
libvhf_md.Rt_to_dm_multi_out.restype = ctypes.c_int

class Int3c2eOpt:
    def __init__(self, components, auxmol):
        self.components = components
        self.auxmol = auxmol
        self.mol = None
        self.component_names = None
        self.component_opts = None
        self.int3c2e_envs = None

    def build(self, cutoff=1e-12):
        component_names = list(self.components)
        mols = [self.components[t] for t in component_names]
        log = logger.new_logger(mols[0])
        cput0 = log.init_timer()
        mol = mols[0]
        for mol1 in mols[1:]:
            mol = gto.conc_mol(mol, mol1)
        # One sorted molecule provides a shared angular shell-pair schedule.
        # Cross-component AO pairs are removed because each density matrix is
        # defined in one distinguishable component space.
        self.mol = mol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=1e200)
        # very high angular momentum basis are processed on CPU
        lmax = mol.uniq_l_ctr[:,0].max()
        assert lmax <= LMAX

        component_opts = {}
        orig_shell_component = []
        orig_shell_local = []
        # component_opts[t] supplies t's sorted AO transformation and matrix size.
        for ic, (t, mol_t) in enumerate(zip(component_names, mols)):
            opt_t = df_j_engine_3c2e.Int3c2eOpt(mol_t, self.auxmol)
            opt_t.mol = SortedGTO.from_mol(
                mol_t, decontract=True, diffuse_cutoff=1e200)
            component_opts[t] = opt_t
            orig_shell_component.extend([ic] * opt_t.mol.nbas)
            orig_shell_local.extend(range(opt_t.mol.nbas))
        # shell_component and shell_local map each sorted combined shell to its
        # component and original component-local shell index.
        orig_shell_component = np.asarray(orig_shell_component, dtype=np.int32)
        orig_shell_local = np.asarray(orig_shell_local, dtype=np.int32)
        shell_component = orig_shell_component[mol.sorted_idx]
        shell_local = orig_shell_local[mol.sorted_idx]

        # sorted_local_shell maps original component shell indices to the
        # independently sorted component shell indices used for local AO ranges.
        sorted_local_shell = {}
        for t, opt_t in component_opts.items():
            inv_sorted = np.empty_like(opt_t.mol.sorted_idx)
            inv_sorted[opt_t.mol.sorted_idx] = np.arange(opt_t.mol.nbas)
            sorted_local_shell[t] = inv_sorted

        _atm = cp.array(mol._atm)
        _bas = cp.array(mol._bas)
        _env = cp.array(_scale_sp_ctr_coeff(mol))
        ao_loc = cp.array(mol.ao_loc)
        rys_envs = RysIntEnvVars.new(mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)
        self.bas_pair_cache = bas_pair_cache = df_j_engine_3c2e._cache_q_cond_and_non0pairs(
            mol, rys_envs, cutoff)
        log.timer('Initialize q_cond', *cput0)

        auxmol = self.auxmol = SortedGTO.from_mol(
            self.auxmol, decontract=True, diffuse_cutoff=1e200)
        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off
        self._atm = _atm_cpu
        self._bas = _bas_cpu
        self._env = _env_cpu

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(mol.ao_loc, auxmol.ao_loc_nr(cart=True))
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)
        self.int3c2e_envs = RysIntEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)

        # Keep only same-component shell pairs and group them by angular block.
        pair_lst = []
        # pair_component[p] owns shell-pair job p in pair_lst.
        pair_component = []
        # shl_pair_offsets delimit angular blocks in pair_lst.
        shl_pair_offsets = [0]
        p1 = 0
        nbas = mol.nbas
        for key in sorted(bas_pair_cache):
            pair = bas_pair_cache[key][0].get()
            if pair.size == 0:
                continue
            ish = pair // nbas
            jsh = pair % nbas
            same_component = shell_component[ish] == shell_component[jsh]
            if not np.any(same_component):
                continue
            pair = pair[same_component]
            ish = ish[same_component]
            component = shell_component[ish]
            for ic in range(len(component_names)):
                pair_t = pair[component == ic]
                if pair_t.size == 0:
                    continue
                pair_lst.append(pair_t)
                pair_component.append(np.full(pair_t.size, ic, dtype=np.int32))
                p1 += pair_t.size
            shl_pair_offsets.append(p1)
        pair_lst = np.hstack(pair_lst).astype(np.uint32)
        self.shl_pair_idx = pair_lst
        self.pair_component = np.hstack(pair_component).astype(np.int32)

        # local_ao_loc[ish] is shell ish's AO offset in its component matrix.
        local_ao_loc = np.empty(mol.nbas, dtype=np.int32)
        component_nao = np.empty(len(component_names), dtype=np.int32)
        for ic, t in enumerate(component_names):
            opt_t = component_opts[t]
            component_nao[ic] = opt_t.mol.nao
            inv_sorted = sorted_local_shell[t]
            idx = np.where(shell_component == ic)[0]
            local_ao_loc[idx] = opt_t.mol.ao_loc[inv_sorted[shell_local[idx]]]

        ls = np.asarray(mol._bas[:,ANG_OF], dtype=np.int32)
        ll = ls[:,None] + ls
        ll = ll.ravel()[pair_lst]
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        self.pair_loc = np.cumsum(np.append(np.int32(0), xyz_size.ravel()), dtype=np.int32)

        self.component_names = component_names
        self.component_opts = component_opts
        self.shell_component = shell_component
        self.local_ao_loc = local_ao_loc
        self.component_nao = component_nao
        self.shl_pair_offsets = np.asarray(shl_pair_offsets, dtype=np.int32)
        return self

    def contract_dm(self, dm, hermi=0):
        if self.int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        naux = auxmol.nao_nr(cart=True)

        # _dms retains the component DMs referenced by dm_ptrs during CUDA calls.
        _dms = []
        n_dm = None
        for t in self.component_names:
            opt_t = self.component_opts[t]
            dm_t = cp.asarray(dm[t])
            ao_loc = opt_t.mol.ao_loc
            nao = ao_loc[-1]
            assert dm_t.shape[-1] == nao, \
                'Requires transforming dm: mol.apply_C_mat_CT(dm)'
            dm_t = dm_t.reshape(-1, nao, nao)
            if hermi != 1:
                dm_t = transpose_sum(dm_t)
            dm_t = cp.asarray(dm_t, order='C')
            _dms.append(dm_t)
            if n_dm is None:
                n_dm = len(dm_t)
            elif len(dm_t) != n_dm:
                raise ValueError('All component density matrices must have the same number of sets')

        nsp_lookup, shm_size = df_j_engine_3c2e._int3c2e_dm_scheme()
        shl_pair_idx = asarray(self.shl_pair_idx, dtype=np.int32)
        pair_ij_offsets = cp.asarray(self.shl_pair_offsets, dtype=np.int32)
        sp_blocks = len(pair_ij_offsets) - 1
        log.debug1('sp_blocks = %d, shm_size = %d B', sp_blocks, shm_size)
        pair_component = cp.asarray(self.pair_component, dtype=np.int32)

        pair_loc = cp.asarray(self.pair_loc, dtype=np.int32)
        omega = mol.omega
        int3c2e_envs = self.int3c2e_envs
        # dm_xyz[:,pair_loc[p]:pair_loc[p+1]] stores the Et transform of shell
        # pair p, read from the component matrix selected by pair_component[p].
        dm_xyz = cp.empty((n_dm, int(pair_loc[-1])))
        dm_ptrs = cp.asarray(np.asarray([dm.data.ptr for dm in _dms], dtype=np.uintp))
        local_ao_loc = cp.asarray(self.local_ao_loc, dtype=np.int32)
        component_id = cp.asarray(self.shell_component, dtype=np.int32)
        component_nao = cp.asarray(self.component_nao, dtype=np.int32)
        err = libvhf_md.dm_to_Rt_multi_in(
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.byref(int3c2e_envs),
            ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(shl_pair_idx)),
            ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(component_id.data.ptr, ctypes.c_void_p),
            ctypes.cast(component_nao.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('dm_to_Rt_multi_in kernel failed')

        vj_aux = cp.zeros((len(self.component_names), n_dm, naux))
        # vj_aux[c,s,P] = sum_pq D_c[s,p,q](pq|P).
        for i_dm in range(n_dm):
            vj_ptrs = [vj_aux[ic,i_dm].data.ptr
                       for ic in range(len(self.component_names))]
            vj_ptrs = cp.asarray(np.asarray(vj_ptrs, dtype=np.uintp))
            err = libvhf_md.contract_int3c2e_dm_multi_out(
                ctypes.cast(vj_ptrs.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_xyz[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(naux),
                ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
                ctypes.c_int(sp_blocks),
                ctypes.c_int(auxmol.nbas),
                ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('contract_int3c2e_dm_multi_out kernel failed')
        if hermi == 1:
            vj_aux *= 2
        if n_dm == 1:
            vj_aux = vj_aux[:,0]
        log.timer_debug1('processing neo contract_int3c2e_dm', *t0)
        return {t: vj_aux[ic] for ic, t in enumerate(self.component_names)}

    def contract_auxvec(self, auxvec, componentwise=False):
        if self.int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        aux_loc = auxmol.ao_loc
        naux = aux_loc[-1]
        # With componentwise, AO pairs of t use auxvec[t][s,P]; otherwise all
        # AO pairs use the same auxvec[s,P].
        if componentwise:
            auxvecs = {}
            n_dm = None
            for t in self.component_names:
                auxvec_t = auxvec[t]
                assert auxvec_t.shape[-1] == naux
                auxvec_t = cp.asarray(auxvec_t.reshape(-1,naux), order='C')
                auxvecs[t] = auxvec_t
                if n_dm is None:
                    n_dm = len(auxvec_t)
                elif len(auxvec_t) != n_dm:
                    raise ValueError('All component auxiliary vectors must have the same number of sets')
        else:
            assert auxvec.shape[-1] == naux
            auxvec = cp.asarray(auxvec.reshape(-1,naux), order='C')
            n_dm = len(auxvec)

        l = auxmol._bas[:,ANG_OF]
        nf3 = (l+1)*(l+2)*(l+3)//6
        aux_xyz_loc = np.asarray(np.append(0, np.cumsum(nf3)), dtype=np.int32)

        nsp_lookup, shm_size = df_j_engine_3c2e._int3c2e_auxvec_scheme()
        nsp_lookup_multi_in, shm_size_multi_in = _int3c2e_auxvec_multi_in_scheme()

        shl_pair_idx = asarray(self.shl_pair_idx, dtype=np.int32)
        pair_ij_offsets = asarray(self.shl_pair_offsets, dtype=np.int32)
        sp_blocks = len(pair_ij_offsets) - 1
        pair_component = cp.asarray(self.pair_component, dtype=np.int32)

        # Split auxbasis into small batches for load balance
        ksh_offsets = []
        k0 = k1 = mol.nbas
        for n in auxmol.l_ctr_counts:
            k0, k1 = k1, k1 + n
            ksh_offsets.append(np.arange(k0, k1, 16, dtype=np.int32))
        ksh_offsets.append(np.int32(k1))
        ksh_offsets = asarray(np.hstack(ksh_offsets, dtype=np.int32))
        ksh_blocks = len(ksh_offsets) - 1

        int3c2e_envs = self.int3c2e_envs
        aux_loc = asarray(aux_loc)
        aux_xyz_size = int(aux_xyz_loc[-1])
        aux_xyz_loc = asarray(aux_xyz_loc)
        pair_loc = asarray(self.pair_loc)
        omega = mol.omega

        log.debug1('sp_blocks = %d, ksh_blocks = %d, shm_size = %d B',
                   sp_blocks, ksh_blocks, shm_size)

        # vj[t][s] is a dense (nao_t,nao_t) target-component matrix.
        vj = {}
        for t in self.component_names:
            opt_t = self.component_opts[t]
            vj[t] = cp.zeros((n_dm, opt_t.mol.nao, opt_t.mol.nao))
        vj_xyz = cp.zeros((n_dm, int(self.pair_loc[-1])))
        if componentwise:
            aux_xyzs = {t: cp.empty((n_dm, aux_xyz_size))
                        for t in self.component_names}
        else:
            aux_xyz = cp.empty(aux_xyz_size)
        local_ao_loc = cp.asarray(self.local_ao_loc, dtype=np.int32)
        component_id = cp.asarray(self.shell_component, dtype=np.int32)
        component_nao = cp.asarray(self.component_nao, dtype=np.int32)

        for i_dm in range(n_dm):
            if componentwise:
                # aux_xyzs[t][s] is the Rt transform of auxvec[t][s]; aux_ptrs
                # lets each shell-pair job select its target component's vector.
                for t in self.component_names:
                    libvhf_md.aux_to_Rt(
                        ctypes.cast(aux_xyzs[t][i_dm].data.ptr, ctypes.c_void_p),
                        ctypes.cast(auxvecs[t][i_dm].data.ptr, ctypes.c_void_p),
                        ctypes.byref(int3c2e_envs),
                        ctypes.cast(aux_loc.data.ptr, ctypes.c_void_p),
                        ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(auxmol.nbas))
                aux_ptrs = [aux_xyzs[t][i_dm].data.ptr for t in self.component_names]
                aux_ptrs = cp.asarray(np.asarray(aux_ptrs, dtype=np.uintp))
                err = libvhf_md.contract_int3c2e_auxvec_multi_in(
                    ctypes.cast(vj_xyz[i_dm].data.ptr, ctypes.c_void_p),
                    ctypes.cast(aux_ptrs.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(1), ctypes.c_int(naux),
                    ctypes.byref(int3c2e_envs),
                    ctypes.c_int(shm_size_multi_in),
                    ctypes.c_int(sp_blocks),
                    ctypes.c_int(ksh_blocks),
                    ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(nsp_lookup_multi_in.data.ptr,
                                ctypes.c_void_p),
                    ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError('contract_int3c2e_auxvec_multi_in kernel failed')
            else:
                # One Rt-transformed auxvec is shared by all shell-pair jobs.
                libvhf_md.aux_to_Rt(
                    ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(auxvec[i_dm].data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs),
                    ctypes.cast(aux_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(auxmol.nbas))
                err = libvhf_md.contract_int3c2e_auxvec(
                    ctypes.cast(vj_xyz[i_dm].data.ptr, ctypes.c_void_p),
                    ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(1), ctypes.c_int(naux),
                    ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size),
                    ctypes.c_int(sp_blocks),
                    ctypes.c_int(ksh_blocks),
                    ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(aux_xyz_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(omega))
                if err != 0:
                    raise RuntimeError('contract_int3c2e_auxvec kernel failed')

            vj_ptrs = [vj[t][i_dm].data.ptr for t in self.component_names]
            vj_ptrs = cp.asarray(np.asarray(vj_ptrs, dtype=np.uintp))
            err = libvhf_md.Rt_to_dm_multi_out(
                ctypes.cast(vj_ptrs.data.ptr, ctypes.c_void_p),
                ctypes.cast(vj_xyz[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.byref(int3c2e_envs),
                ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(self.shl_pair_idx)),
                ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(component_id.data.ptr, ctypes.c_void_p),
                ctypes.cast(component_nao.data.ptr, ctypes.c_void_p))
            if err != 0:
                raise RuntimeError('Rt_to_dm_multi_out kernel failed')
        for t in self.component_names:
            vj[t] = transpose_sum(vj[t])
            if n_dm == 1:
                vj[t] = vj[t][0]
        log.timer_debug1('processing neo contract_int3c2e_auxvec', *t0)
        return vj


def _int3c2e_auxvec_multi_in_scheme():
    # Different shell-pair lanes may select different auxvecs, so each lane
    # needs its own nf3k-element auxiliary cache.
    li = np.arange(LMAX*2+1)
    lk = np.arange(L_AUX_MAX+1)
    nf3k = (L_AUX_MAX+1)*(L_AUX_MAX+2)*(L_AUX_MAX+3)//6
    order = li[:,None] + lk
    nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
    Rt_swap_size = np.array([35, 35, 35, 35, 35, 21, 21])
    Rt_stride = (nf3ijkl + Rt_swap_size-1) // Rt_swap_size
    nfij = (li + 1) * (li + 2) * (li + 3) // 6
    IJ_SIZE = np.array([35, 21, 15, 11, 8, 8, 8])
    Rt_stride_min = (nfij[:,None] + IJ_SIZE-1) // IJ_SIZE
    Rt_stride = np.where(Rt_stride > Rt_stride_min, Rt_stride, Rt_stride_min)

    nsp_max = THREADS // Rt_stride
    unit = order+1 + nf3ijkl + nf3k
    nsp_per_block = SHM_SIZE //(unit*8)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    nsp_per_block = _nearest_power2(nsp_per_block)
    nsp_per_block[nsp_per_block>THREADS] = THREADS

    shm_size = (nsp_per_block * unit).max() * 8
    nsp_lookup = asarray(nsp_per_block, dtype=np.int32)
    return nsp_lookup, shm_size
