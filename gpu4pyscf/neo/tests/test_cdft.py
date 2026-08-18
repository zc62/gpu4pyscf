import unittest
import numpy
from pyscf import neo
from gpu4pyscf import neo as gpu_neo


def _assert_cdft_position_constraint(test_case, mf, atol=1e-6):
    dm = mf.make_rdm1()
    for t, comp in mf.components.items():
        if t.startswith('n'):
            dev = numpy.einsum('xij,ji->x', comp.int1e_r, dm[t].get())
            numpy.testing.assert_allclose(dev, numpy.zeros_like(dev), atol=atol)


def setUpModule():
    global mol
    mol = neo.M(atom='H 0 0 0; C 0 0 1.064; N 0 0 2.220',
                basis='sto-3g', nuc_basis='pb4d', quantum_nuc=[0],
                verbose=0, output='/dev/null')


def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_cdft_noepc(self):
        mf_cpu = neo.CDFT(mol, xc='PBE', epc=None)
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-11
        mf_cpu.max_cycle = 100
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.CDFT(mol, xc='PBE', epc=None)
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        mf_gpu.max_cycle = mf_cpu.max_cycle
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)
        numpy.testing.assert_allclose(mf_gpu.f[0], mf_cpu.f[0], atol=1e-7)
        _assert_cdft_position_constraint(self, mf_gpu)
        numpy.testing.assert_allclose(mf_gpu.dip_moment(verbose=0),
                                      mf_cpu.dip_moment(verbose=0),
                                      atol=1e-6)

    def test_cdft_epc17_2(self):
        mf_cpu = neo.CDFT(mol, xc='PBE', epc='17-2')
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-11
        mf_cpu.max_cycle = 100
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.CDFT(mol, xc='PBE', epc='17-2')
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        mf_gpu.max_cycle = mf_cpu.max_cycle
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)
        numpy.testing.assert_allclose(mf_gpu.f[0], mf_cpu.f[0], atol=2e-7)
        _assert_cdft_position_constraint(self, mf_gpu)

    def test_to_gpu(self):
        mf_cpu = neo.CDFT(mol, xc='PBE', epc=None)
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-9
        mf_cpu.max_cycle = 50
        e_cpu = mf_cpu.kernel()

        mf = neo.CDFT(mol, xc='PBE', epc=None).to_gpu()
        self.assertIsInstance(mf, gpu_neo.CDFT)
        mf.components['e'].grids.level = 1
        mf.conv_tol = mf_cpu.conv_tol
        mf.max_cycle = mf_cpu.max_cycle
        self.assertAlmostEqual(mf.kernel(), e_cpu, 8)

    def test_to_cpu(self):
        mf = gpu_neo.CDFT(mol, xc='PBE', epc=None)
        mf_cpu = mf.to_cpu()
        self.assertIsInstance(mf_cpu, neo.CDFT)


if __name__ == '__main__':
    print('Full Tests for gpu4pyscf.neo.cdft')
    unittest.main()
