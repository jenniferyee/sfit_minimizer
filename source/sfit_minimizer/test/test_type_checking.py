"""
Check that the type-checking works as expected.
"""

import unittest

import numpy as np

from sfit_minimizer.sfit_classes import SFitFunction


class TestType4SFit(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.dates = range(0, self.n)
        self.values = range(0, self.n)
        self.errs = range(0, self.n)
        self.data = np.array([self.dates, self.values, self.errs]).transpose()


# data: np.array, dimensions
class TestDataType(TestType4SFit):
    def test_data_type_1(self):
        # wrong type
        with self.assertRaises(TypeError):
            sfit_1 = SFitFunction(data=[self.dates, self.values, self.errs])

    def test_data_type_2(self):
        # wrong shape
        with self.assertRaises(ValueError):
            sfit_2 = SFitFunction(data=np.array([self.dates, self.values]))

    # correct behavior
    def test_data_type_3(self):
        sfit_3 = SFitFunction(data=self.data)


# theta: type, dimensions
class TestThetaType(TestType4SFit):
    def setUp(self):
        TestType4SFit.setUp(self)
        self.func = SFitFunction(data=self.data)

    def test_theta_type_1(self):
        # correct behavior for no theta value
        self.func.theta = None

    def test_theta_type_2(self):
        # List
        self.func.theta = [1, 2.1]

    def test_theta_type_3(self):
        # np.array
        self.func.theta = np.array([1, 2.1])

    def test_theta_type_4(self):
        # Tuple should fail
        with self.assertRaises(TypeError):
            self.func.theta = (1, 2.1)


# ymod: type, dimensions
class TestYModType(TestType4SFit):
    def setUp(self):
        TestType4SFit.setUp(self)
        self.func = SFitFunction(data=self.data)

    def test_ymod_1(self):
        # works correctly
        self.func.ymod = np.arange(0, self.n)

    def test_ymod_2(self):
        # wrong type
        with self.assertRaises(TypeError):
            self.func.ymod = range(0, self.n)

    def test_ymod_3(self):
        # wrong size
        with self.assertRaises(ValueError):
            self.func.ymod = np.arange(0, self.n + 1)


# res: type, dimensions
class TestResType(TestType4SFit):
    def setUp(self):
        TestType4SFit.setUp(self)
        self.func = SFitFunction(data=self.data)

    def test_res_1(self):
        # works correctly
        self.func.residuals = np.arange(0, self.n)

    def test_res_2(self):
        # wrong type
        with self.assertRaises(TypeError):
            self.func.residuals = range(0, self.n)

    def test_res_3(self):
        # wrong size
        with self.assertRaises(ValueError):
            self.func.residuals = np.arange(0, self.n + 1)


# df: type, dimensions
class TestDFType(TestType4SFit):
    def setUp(self):
        TestType4SFit.setUp(self)
        self.func = SFitFunction(data=self.data, theta=[0, 1])

    def test_df_1(self):
        # works correctly
        self.func.df = np.zeros((2, self.n))

    def test_df_2(self):
        # wrong type
        with self.assertRaises(TypeError):
            self.func.df = [range(0, self.n), range(0, self.n)]

    def test_df_3(self):
        # wrong size
        with self.assertRaises(ValueError):
            self.func.df = np.zeros((2, self.n + 1))

    def test_df_4(self):
        # wrong size
        with self.assertRaises(ValueError):
            self.func.df = np.zeros((3, self.n + 1))
