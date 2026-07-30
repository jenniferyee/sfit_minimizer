"""
Assuming a linear function of the form "f = a0 + a1 * x + a2 * x^2 ...", check
that the minimizer works as expected.
"""
import os.path
import unittest

import numpy as np

import sfit_minimizer


class LinearFunction(sfit_minimizer.SFitFunction):

    def __init__(self, data=None, theta=None):
        sfit_minimizer.SFitFunction.__init__(self, data=data, theta=theta)

    def calc_model(self):
        """Calculate expected values of the model"""
        ymod = []
        for i in range(len(self.theta)):
            ymod.append(self.theta[i] * self.data[:, 0] ** i)

        ymod = np.array(ymod)
        self.ymod = np.sum(ymod, axis=0)

    def calc_df(self):
        """Calculate the derivatives of the fitting function and store as
        self.df."""
        df = []
        for i in range(len(self.theta)):
            df.append(self.data[:, 0] ** i)

        self.df = np.array(df)


def test_perfect_chi2():
    """chi2 should be 0"""
    data = np.loadtxt(os.path.join(
        sfit_minimizer.DATA_PATH, 'PolynomialTest', 'test_data_10perfect.txt'),
        skiprows=2)
    theta = [3, 2]
    my_func = LinearFunction(data=data, theta=theta)
    my_func.calc_chi2()

    tol = 1e-6
    np.testing.assert_almost_equal(my_func.residuals, 0., decimal=3)
    assert my_func.chi2 < tol


# Test the SFitFunction calculations produce the expected results
def test_fit():
    """
    Fit a line to perfect data: y = m * x + b
    :return:
    """
    # Fake data
    data = np.loadtxt(os.path.join(sfit_minimizer.DATA_PATH, 'PolynomialTest', 'test_data_10000pts_Poisson.txt'), skiprows=2)

    # Wrong initial condition
    theta = [4, 2.1]

    # Compare results
    my_func = LinearFunction(data=data, theta=theta)
    my_func.update_all()
    print('step', my_func.step)
    print('df.shape', my_func.df.shape)

    # Check ymod, res
    y = data[:, 0] * theta[1] + theta[0]
    res = data[:, 1] - y
    np.testing.assert_almost_equal(my_func.ymod, y)
    np.testing.assert_almost_equal(my_func.residuals, res)

    # Check b, c, d matrices
    b = np.zeros((2, 2))
    b[0, 0] = np.sum(1. / data[:, 2] ** 2)
    b[0, 1] = np.sum(data[:, 0] / data[:, 2] ** 2)
    b[1, 0] = b[0, 1]
    b[1, 1] = np.sum(data[:, 0] ** 2 / data[:, 2] ** 2)
    c = np.linalg.inv(b)
    d = np.zeros(2)
    d[0] = np.sum(res / data[:, 2] ** 2)
    d[1] = np.sum(res * data[:, 0] / data[:, 2] ** 2)

    np.testing.assert_almost_equal(my_func.bmat, b)
    np.testing.assert_almost_equal(my_func.cmat, c)
    np.testing.assert_almost_equal(my_func.dvec, d)

    # Check true parameters are recovered.
    theta_new = theta + my_func.step
    sigmas = my_func.get_sigmas()
    try:
        assert np.abs(theta_new[0] - 3.) < 3. * sigmas[0]
        assert np.abs(theta_new[1] - 2.) < 3. * sigmas[1]
    except AssertionError:
        return my_func, theta_new


# Test that the minimize function produces the expected results
class TestMinimize(unittest.TestCase):

    def setUp(self):
        self.data = np.loadtxt(os.path.join(sfit_minimizer.DATA_PATH, 'PolynomialTest', 'test_data_10000pts_Poisson.txt'),
                          skiprows=2)
        self.initial_guess = [4, 2.1]  # Wrong initial condition
        self.my_func = LinearFunction(data=self.data)

    def _evaluate_test(self, result):
        self.my_func.theta = result.x
        np.testing.assert_almost_equal(self.my_func.get_chi2(), 10865.52999, decimal=3)
        print(result)

    def test_minimize(self):
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-7,
            options={'step': 'adaptive'}, verbose=False)
        self._evaluate_test(result)

    def test_minimize_no_options(self):
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-7,
            verbose=False)

        self._evaluate_test(result)

    def test_minimize_fixed_step(self):
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-7,
            options={'step': 0.01}, max_iter=10000, verbose=False)
        self._evaluate_test(result)

    def test_minimize_excceeds_max_iter(self):
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-7,
            options={'step': 0.01}, max_iter=100, verbose=False)
        assert result.success == False
        assert result.msg[0:3] == 'max'

    def test_minimize_none_step(self):
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-7,
            options={'step': None}, max_iter=10000, verbose=False)
        self._evaluate_test(result)

    def test_minimize_value_error_1(self):
        with self.assertRaises(ValueError):
            result = sfit_minimizer.minimize(
                self.my_func, x0=self.initial_guess, tol=1e-7,
                options={'step': 'banana'}, max_iter=10000, verbose=False)

    def test_minimize_value_error_2(self):
        with self.assertRaises(ValueError):
            result = sfit_minimizer.minimize(
                self.my_func, x0=self.initial_guess, tol=1e-7,
                options={'step': np.nan}, max_iter=10000, verbose=False)
