import unittest

import numpy
from pyscf import neo, qmmm
from pyscf.neo import grad as cpu_grad

from gpu4pyscf import neo as gpu_neo


def setUpModule():
    global mol, mol_multi, mol_epc, mol_ecp
    mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                nuc_basis='pb4d', quantum_nuc=[0],
                verbose=0, output='/dev/null')
    mol_multi = neo.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                      basis='sto-3g', nuc_basis='pb4d', quantum_nuc=['H'],
                      verbose=0, output='/dev/null')
    mol_epc = neo.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='6-31g', nuc_basis='pb4d', quantum_nuc=[1],
                    verbose=0, output='/dev/null')
    mol_ecp = neo.M(atom='Na 0 0 0; H 0 0 3.0',
                    basis={'Na': 'lanl2dz', 'H': 'sto-3g'},
                    ecp={'Na': 'lanl2dz'}, nuc_basis='pb4d',
                    quantum_nuc=[1], unit='Bohr',
                    verbose=0, output='/dev/null')


def tearDownModule():
    global mol, mol_multi, mol_epc, mol_ecp
    mol.stdout.close()
    mol_multi.stdout.close()
    mol_epc.stdout.close()
    mol_ecp.stdout.close()
    del mol
    del mol_multi
    del mol_epc
    del mol_ecp


def run_cpu(mf, grids_level=1, conv_tol=1e-9):
    if hasattr(mf.components['e'], 'grids'):
        mf.components['e'].grids.level = grids_level
    mf.conv_tol = conv_tol
    mf.kernel()
    return cpu_grad.Gradients(mf).kernel()


def run_gpu(mf, grids_level=1, conv_tol=1e-9):
    if hasattr(mf.components['e'], 'grids'):
        mf.components['e'].grids.level = grids_level
    mf.conv_tol = conv_tol
    mf.kernel()
    return mf.Gradients().kernel()


class KnownValues(unittest.TestCase):
    def test_hf_grad(self):
        g_cpu = run_cpu(neo.HF(mol))
        g_gpu = run_gpu(gpu_neo.HF(mol))
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=1e-6)

    def test_hf_multiple_quantum_nuclei_grad(self):
        g_cpu = run_cpu(neo.HF(mol_multi))
        g_gpu = run_gpu(gpu_neo.HF(mol_multi))
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=1e-6)

    def test_hf_ecp_grad(self):
        g_cpu = run_cpu(neo.HF(mol_ecp), conv_tol=1e-11)
        g_gpu = run_gpu(gpu_neo.HF(mol_ecp), conv_tol=1e-11)
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=1e-6)

    def test_ks_grad(self):
        g_cpu = run_cpu(neo.KS(mol, xc='PBE', epc=None))
        g_gpu = run_gpu(gpu_neo.KS(mol, xc='PBE', epc=None))
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=1e-6)

    def test_ks_epc_grad(self):
        g_cpu = run_cpu(neo.KS(mol_epc, xc='PBE', epc='17-2'),
                        grids_level=2, conv_tol=1e-10)
        g_gpu = run_gpu(gpu_neo.KS(mol_epc, xc='PBE', epc='17-2'),
                        grids_level=2, conv_tol=1e-10)
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=2e-6)

    def test_hf_qmmm(self):
        mm_coords = numpy.array([[0.8, 0.2, 1.5]])
        mm_charges = numpy.array([0.3])

        try:
            mf_cpu = neo.HF(mol)
            mf_cpu.mol.mm_mol = qmmm.mm_mole.create_mm_mol(
                mm_coords, mm_charges, unit='Bohr')
            mf_cpu.conv_tol = 1e-10
            e_cpu = mf_cpu.kernel()
            gobj_cpu = cpu_grad.Gradients(mf_cpu)
            g_cpu = gobj_cpu.kernel()
            gm_cpu = gobj_cpu.grad_mm()

            mf_gpu = gpu_neo.HF(mol)
            mf_gpu.mol.mm_mol = qmmm.mm_mole.create_mm_mol(
                mm_coords, mm_charges, unit='Bohr')
            mf_gpu.conv_tol = 1e-10
            e_gpu = mf_gpu.kernel()
            gobj_gpu = mf_gpu.Gradients()
            g_gpu = gobj_gpu.kernel()
            gm_gpu = gobj_gpu.grad_mm()

            self.assertTrue(mf_cpu.converged)
            self.assertTrue(mf_gpu.converged)
            self.assertAlmostEqual(e_gpu, e_cpu, 9)
            numpy.testing.assert_allclose(g_gpu, g_cpu, atol=1e-6)
            numpy.testing.assert_allclose(gm_gpu, gm_cpu, atol=1e-6)
        finally:
            mol.mm_mol = None


if __name__ == '__main__':
    print('Full Tests for NEO gradients')
    unittest.main()
