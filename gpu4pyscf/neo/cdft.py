import cupy
import numpy
import scipy.optimize
from pyscf import scf as scf_cpu
from pyscf.neo import cdft as cdft_cpu
from pyscf.neo import hf as hf_cpu
from pyscf.neo import ks as ks_cpu
from gpu4pyscf import scf
from gpu4pyscf.lib import utils
from gpu4pyscf.neo import hf, ks


def _get_mo_coeff_occ(mf, fock, s1e):
    mo_energy, mo_coeff = scf_cpu.hf.eig(fock, s1e)
    verbose = mf.verbose
    nnuc = mf.mol.nnuc
    try:
        # Temporarily disable the verbose output in get_occ
        mf.verbose = 0
        # Temporarily set nnuc=1 for expectation position constraint
        # and restore nnuc before returning to the physical SCF density.
        mf.mol.nnuc = 1.0
        mo_occ = hf_cpu.ComponentSCF.get_occ(mf, mo_energy, mo_coeff)
    finally:
        mf.mol.nnuc = nnuc
        mf.verbose = verbose
    return mo_coeff, mo_occ


def solve_constraint(mf, fock0, s1e=None, f_lagrange_guess=None):
    '''Solve the Kohn-Sham equation with position constraint
        [H + f_lagrange * (r - R)] y = e y, <y|r - R|y> = 0.
    '''
    if s1e is None:
        s1e = hf._to_cpu(mf.get_ovlp())
    if f_lagrange_guess is None:
        f_lagrange_guess = numpy.zeros(mf.int1e_r.shape[0])

    if mf.int1e_r_symm is not None:
        raise NotImplementedError('Symmetry adapted CDFT position constraint '
                                  'is not implemented')

    def position_deviation(f_lagrange):
        '''Calculate position deviation from the Kohn-Sham orbital with
        frozen unconstrained NEO Fock and provided Lagrange multiplier'''
        # Get Fock matrix with constraint
        fock = fock0 + numpy.einsum('xij,x->ij', mf.int1e_r, f_lagrange)

        # Calculate expectation position deviation
        mo_coeff, mo_occ = _get_mo_coeff_occ(mf, fock, s1e)
        dm = scf_cpu.hf.make_rdm1(mo_coeff, mo_occ)
        deviation = numpy.einsum('xij,ji->x', mf.int1e_r, dm)
        return deviation

    #opt = scipy.optimize.root(position_deviation, f_lagrange_guess, method='hybr')
    opt = scipy.optimize.least_squares(position_deviation, f_lagrange_guess, gtol=1e-15)
    return opt


class CDFT(ks.KS):
    _keys = ks.KS._keys.union({'f'})

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.f = numpy.zeros((mol.natm, 3))
        self._setup_position_matrices()

    def _setup_position_matrices(self):
        '''Set up position matrices for each quantum nucleus for constraint'''
        for t, comp in self.components.items():
            if t.startswith('n'):
                if comp.mol.symmetry:
                    raise NotImplementedError('Symmetry adapted CDFT position '
                                              'constraint is not implemented')
                comp.nuclear_expect_position = \
                    comp.mol.atom_coord(comp.mol.atom_index)
                s1e = hf._to_cpu(comp.get_ovlp())
                r = comp.mol.intor_symmetric('int1e_r', comp=3)
                r0 = numpy.asarray([comp.nuclear_expect_position[i] * s1e
                                    for i in range(3)])
                comp.int1e_r = r - r0
                comp.int1e_r_symm = None

    def get_fock_add_cdft(self):
        '''Get additional Fock terms from constraints'''
        f_add = {}
        for t, comp in self.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                f_add[t] = cupy.asarray(numpy.einsum('xij,x->ij', comp.int1e_r, self.f[ia]))
        return f_add

    dip_moment = cdft_cpu.CDFT.dip_moment

    def reset(self, mol=None):
        super().reset(mol=mol)
        self.f = numpy.zeros((self.mol.natm, 3))
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
        if key == 'f':
            setattr(out, key, numpy.asarray(val))
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
