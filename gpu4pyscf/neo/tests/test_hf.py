import unittest

import numpy
from pyscf import neo

import gpu4pyscf.neo as gpu_neo
from gpu4pyscf import scf as gpu_scf


def setUpModule():
    global mol_h, mol
    mol_h = neo.M(atom='H 0 0 0', basis='sto-3g', nuc_basis='pb4d',
                  quantum_nuc=[0], spin=1, verbose=0,
                  output='/dev/null')
    mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                nuc_basis='pb4d', quantum_nuc=[0],
                verbose=0, output='/dev/null')


def tearDownModule():
    global mol_h, mol
    mol_h.stdout.close()
    mol.stdout.close()
    del mol_h, mol


class KnownValues(unittest.TestCase):
    def test_scanner_spin(self):
        mol1 = neo.M(atom='H 0 0 0', basis='sto-3g', nuc_basis='pb4d',
                     quantum_nuc=[0], spin=1, verbose=0)
        mol = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                    nuc_basis='pb4d', quantum_nuc=[0], verbose=0)
        mol2 = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                     nuc_basis='pb4d', quantum_nuc=[0], charge=1, spin=1, verbose=0)
        mf = gpu_neo.HF(mol1)
        mf.conv_tol = 1e-10
        scanner = mf.as_scanner()
        scanner(mol1)
        mf_e = scanner.components['e']
        for mol_test, unrestricted in ((mol2, False), (mol, False), (mol, True)):
            scanner.unrestricted = unrestricted
            mf_ref = gpu_neo.HF(mol_test, unrestricted=unrestricted)
            mf_ref.conv_tol = 1e-10
            self.assertAlmostEqual(scanner(mol_test), mf_ref.scf(), 8)
            self.assertEqual(isinstance(scanner.components['e'], gpu_scf.uhf.UHF),
                             unrestricted or mol_test.spin != 0)
            if mol_test is mol2:
                self.assertIs(scanner.components['e'], mf_e)

    def test_to_gpu_kernel(self):
        mf_cpu = neo.HF(mol)
        mf_cpu.conv_tol = 1e-11
        e_cpu = mf_cpu.kernel()

        mf_gpu = neo.HF(mol).to_gpu()
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 9)

    def test_direct_gpu_hf_kernel(self):
        mf_cpu = neo.HF(mol)
        mf_cpu.conv_tol = 1e-11
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.HF(mol)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 9)

    def test_hf1e_h_atom(self):
        mf_cpu = neo.HF(mol_h)
        mf_cpu.conv_tol = 1e-11
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.HF(mol_h)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 9)

    def test_to_cpu(self):
        mf_gpu = neo.HF(mol).to_gpu()
        mf_cpu = mf_gpu.to_cpu()

        self.assertIsInstance(mf_cpu, neo.hf.HF)
        self.assertEqual(sorted(mf_cpu.components), sorted(mf_gpu.components))
        for comp in mf_cpu.components.values():
            self.assertFalse(comp.__class__.__module__.startswith('gpu4pyscf'))
        self.assertIsInstance(mf_cpu.get_hcore()['e'], numpy.ndarray)

    def test_dip_moment(self):
        mf_cpu = neo.HF(mol)
        mf_cpu.conv_tol = 1e-11
        mf_cpu.kernel()

        mf_gpu = gpu_neo.HF(mol)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        numpy.testing.assert_allclose(mf_gpu.dip_moment(verbose=0),
                                      mf_cpu.dip_moment(verbose=0),
                                      atol=1e-7)


if __name__ == '__main__':
    print('Full Tests for NEO HF')
    unittest.main()
