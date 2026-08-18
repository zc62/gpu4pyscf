import unittest

import numpy
from pyscf import neo

import gpu4pyscf.neo as gpu_neo


def setUpModule():
    global mol
    mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                nuc_basis='pb4d', quantum_nuc=[0], verbose=0,
                output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_direct_gpu_no_epc(self):
        mf_cpu = neo.KS(mol, xc='PBE', epc=None)
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='PBE', epc=None)
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_direct_gpu_epc17_2(self):
        mf_cpu = neo.KS(mol, xc='PBE', epc='17-2')
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='PBE', epc='17-2')
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_to_gpu_epc(self):
        mf_cpu = neo.KS(mol, xc='PBE', epc='17-2')
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()

        mf_gpu = neo.KS(mol, xc='PBE', epc='17-2').to_gpu()
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_to_cpu(self):
        mf_gpu = neo.KS(mol, xc='PBE', epc='17-2').to_gpu()
        mf_cpu = mf_gpu.to_cpu()

        self.assertIsInstance(mf_cpu, neo.ks.KS)
        self.assertEqual(sorted(mf_cpu.components), sorted(mf_gpu.components))
        self.assertEqual(mf_cpu.xc_e, 'PBE')
        self.assertEqual(mf_cpu.epc, '17-2')
        self.assertIsInstance(mf_cpu.get_hcore()['e'], numpy.ndarray)


if __name__ == '__main__':
    print('Full Tests for NEO KS')
    unittest.main()
