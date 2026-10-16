#!/usr/bin/env python

"""Unit tests for the three-dimensional PDF calculator."""

import pickle
import tempfile
import unittest

import numpy
from testutils import loadDiffPyStructure

from diffpy.srreal.parallel import createParallelCalculator
from diffpy.srreal.pdfcalculator import PDF3DCalculator


class TestPDF3DCalculator(unittest.TestCase):

    nickel = None

    @classmethod
    def setUpClass(cls):
        cls.nickel = loadDiffPyStructure("Ni.stru")

    def newCalculator(self, **kwargs):
        settings = dict(
            rmax=3.0,
            gridstep=0.3,
            qmax=0,
            usecqwindow3d=False,
            applyrho0background3d=False,
        )
        settings.update(kwargs)
        return PDF3DCalculator(**settings)

    def test_init_and_gridstep(self):
        calc = self.newCalculator()
        self.assertEqual(0.3, calc.gridstep)
        self.assertEqual(calc.gridstep, calc.rstep)

        calc.rstep = 0.2
        self.assertEqual(0.2, calc.gridstep)
        self.assertEqual(calc.gridstep, calc.rstep)

        calc.gridstep = 0.4
        self.assertEqual(0.4, calc.gridstep)
        self.assertEqual(calc.gridstep, calc.rstep)
        with self.assertRaises(ValueError):
            calc.gridstep = 0

    def test_call_returns_3d_pdf(self):
        calc = self.newCalculator()
        result = calc(self.nickel)

        self.assertIsInstance(result, numpy.ndarray)
        self.assertEqual(1, result.ndim)
        self.assertEqual(0, result.size % 4)
        numpy.testing.assert_array_equal(result, calc.pdf3d)

        points = result.reshape(-1, 4)
        self.assertGreater(len(points), 0)
        self.assertTrue(numpy.any(points[:, 3] != 0.0))
        self.assertTrue(numpy.all(numpy.abs(points[:, :3]) <= calc.rmax))

    def test_qwindow_filters_background(self):
        grids = []
        for applybackground in (False, True):
            calc = self.newCalculator(
                rmax=2.0,
                gridstep=0.5,
                qmin=1.0,
                qmax=0,
                usecqwindow3d=True,
                applyrho0background3d=applybackground,
            )
            calc(self.nickel)
            with tempfile.NamedTemporaryFile() as fp:
                calc.exportGrid3DBinary(fp.name, usefloat32=False)
                grids.append(numpy.fromfile(fp.name, dtype=numpy.float64))

        self.assertEqual(9**3, len(grids[0]))
        background = grids[1] - grids[0]
        # qmin filtering must act on the finite-grid background.  Subtracting
        # rho0 after the Q window would leave a spatially constant difference.
        self.assertGreater(numpy.ptp(background), 0.01)

    def test_parallel_matches_serial(self):
        serial = self.newCalculator()(self.nickel)
        parallel = createParallelCalculator(self.newCalculator(), 2, map)
        actual = parallel(self.nickel)
        numpy.testing.assert_allclose(
            actual, serial, rtol=1.0e-12, atol=1.0e-12
        )

    def test_delta_pair_binding_and_pickle(self):
        calc = self.newCalculator()
        calc.setDeltaPairTypes3D("Ni", "*")
        self.assertEqual("Ni", calc.deltapaira3d)
        self.assertEqual("*", calc.deltapairb3d)

        calc(self.nickel)
        restored = pickle.loads(pickle.dumps(calc))
        self.assertEqual(calc.gridstep, restored.gridstep)
        self.assertEqual(calc.deltapaira3d, restored.deltapaira3d)
        numpy.testing.assert_allclose(restored.pdf3d, calc.pdf3d)


if __name__ == "__main__":
    unittest.main()
