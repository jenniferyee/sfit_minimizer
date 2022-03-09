"""
Tests for the MulensModel functions = direct comparisons to Andy's fortran sfit.
"""
import numpy as np
import os.path
import sfit_minimizer
import MulensModel as mm

"""
# Tests I need:

1. fixed blending for a single observatory
2. parallax
3. bad data

"""


mm.utils.MAG_ZEROPOINT = 18.
data_path = os.path.join(sfit_minimizer.DATA_PATH, 'MMTest')


class FortranSFitFile(object):
    """
    Class to parse the comparison file generated by the Fortran version of sfit.
    """

    def __init__(self, filename):
        input_file = open(filename, 'r')
        attr = None
        for line in input_file.readlines():
            str_vec = line.strip().split()
            if str_vec[0] == '#':
                attr = str_vec[-1]
            else:
                if len(str_vec) == 1:
                    value = float(str_vec[0])
                else:
                    value = np.array([float(item) for item in str_vec])

                self.__setattr__(attr, value)

        input_file.close()


class ComparisonTest(object):

    def __init__(self, datafiles=None, comp_dir=None, parameters_to_fit=None,
                 coords=None, verbose=False, fix_blend_flux=None,
                 fix_source_flux=None):

        # Get step size from directory name
        str_vec = comp_dir.split('_')
        self.fac = float(str_vec[2])

        # Read in SFit results
        self.sfit_results = FortranSFitFile(
            os.path.join(data_path, 'Matrices', comp_dir, 'fort.60'))
        self.matrices = []
        for i in range(3):
            self.matrices.append(
                FortranSFitFile(
                    os.path.join(
                        data_path, 'Matrices', comp_dir,
                        'fort.{0}'.format(50 + i + 1))
                ))

        # parameters_to_fit
        self.parameters_to_fit = parameters_to_fit
        self.n_params = len(self.parameters_to_fit)
        self.initial_guess = []
        for i in range(self.n_params):
            self.initial_guess.append(self.matrices[0].a[self._get_index(i)])

        self.datasets = []
        for i, filename in enumerate(datafiles):
            data = mm.MulensData(
                file_name=os.path.join(data_path, filename), phot_fmt='mag')
            self.datasets.append(data)

            flux_guess = [1.0, 0.0]
            if isinstance(fix_blend_flux, list):
                if isinstance(fix_blend_flux[i], float):
                    flux_guess.pop(1)

            if isinstance(fix_source_flux, list):
                if isinstance(fix_source_flux[i], float):
                    flux_guess.pop(0)

            if len(flux_guess) > 0:
                self.initial_guess = np.hstack((self.initial_guess, flux_guess))

        date_threshold = 120000.
        if ((self.datasets[0].time[0] > date_threshold) and
                ('t_0' in self.parameters_to_fit)):
            for i, parameter in enumerate(self.parameters_to_fit):
                if parameter == 't_0':
                    if self.initial_guess[i] < date_threshold:
                        self.initial_guess[i] += 2450000.

        self.n_obs = len(self.datasets)

        self.model = mm.Model(
            {self.parameters_to_fit[i]: self.initial_guess[i] for i in range(
                self.n_params)}, coords=coords)
        if 'pi_E_N' in self.parameters_to_fit:
            self.model.parameters.t_0_par = self.initial_guess[0]

        self.event = mm.Event(datasets=self.datasets, model=self.model)
        if fix_blend_flux is not None:
            for i, item in enumerate(fix_blend_flux):
                if item is not False:
                    self.event.fix_blend_flux[self.datasets[i]] = item

        if fix_source_flux is not None:
            for i, item in enumerate(fix_source_flux):
                if item is not False:
                    self.event.fix_source_flux[self.datasets[i]] = item

        self.my_func = sfit_minimizer.mm_funcs.PSPLFunction(
            self.event, self.parameters_to_fit)

        self.verbose = verbose

        if self.verbose:
            print('dataset, length')
            for i in range(self.n_obs):
                print(i, len(self.datasets[i].time))

            print('initial guess', self.initial_guess)
            print('initial model', self.model)

    def run(self):
        self.test_3_iterations()
        self.test_final_results()

    def test_3_iterations(self):
        # Do comparisons
        # first 3 iterations
        new_guess = self.initial_guess
        for i in range(3):
            print('testing iteration', i)
            self.my_func.update_all(theta0=new_guess, verbose=self.verbose)
            self._compare_vector(
                new_guess, self.matrices[i].a, decimal=2, verbose=self.verbose)
            self.compare_calcs(self.matrices[i])
            new_guess += self.my_func.step * self.fac

    def test_final_results(self):
        # Final results
        result = sfit_minimizer.minimize(
            self.my_func, x0=self.initial_guess, tol=1e-4,
            options={'step': 'adaptive'})

        assert result.success

        self.compare_chi2(self.sfit_results)

        sigmas = self.my_func.get_sigmas()
        for i, value0 in enumerate(self.my_func.theta):
            # Step
            np.testing.assert_almost_equal(self.my_func.step[i], 0.0, decimal=3)

            # Value
            index = self._get_index(i)
            if i == 0:
                value = self._t0_correction(value0)
            else:
                value = value0

            assert(
                np.abs(value - self.sfit_results.a[index]) < 0.05 * sigmas[i])

        # sigmas
        self._compare_vector(
           sigmas, self.sfit_results.s, decimal=2)

    def compare_calcs(self, sfit_matrix):
        # chi2
        self.compare_chi2(sfit_matrix)

        # b matrix
        n_elements = int(np.sqrt(len(sfit_matrix.b)))
        shape = (n_elements, n_elements)
        bmat = sfit_matrix.b.reshape(shape)
        self._compare_matrix(
            self.my_func.bmat, bmat, decimal=2, verbose=self.verbose)

        # d vector
        self._compare_vector(
            self.my_func.dvec, sfit_matrix.d, verbose=self.verbose)

        # c matrix
        cmat = sfit_matrix.c.reshape(shape)
        self._compare_matrix(
            self.my_func.cmat, cmat, decimal=2, verbose=self.verbose)

        # step
        self._compare_vector(
            self.my_func.step, sfit_matrix.da, verbose=self.verbose)

    def compare_chi2(self, sfit_matrix):
        if isinstance(sfit_matrix.chi2, (list, np.ndarray)):
            if len(self.datasets) != len(sfit_matrix.chi2):
                raise ValueError(
                    'Number of sfit chi2s != number of datasets:' +
                    '{0}, {1}'.format(
                        len(self.datasets), len(sfit_matrix.chi2)))

        else:
            sfit_matrix.chi2 = [sfit_matrix.chi2]

        for i in range(len(self.datasets)):
            np.testing.assert_almost_equal(
                np.sum(sfit_matrix.chi2[i]),
                self.my_func.event.get_chi2_for_dataset(i), decimal=2)

    def _get_index(self, i):
        """
        Fortran SFit includes a bunch of properties I'm not using
        (e.g., gammas, fsee).

        sfit count: 0-3: t0, u0, tE, rho
                    4-6: gammai,v,h
                    7-8: piEn,e
                    9-11: fs_i, fb_i, fsee_i
        """

        if i < self.n_params:
            # ulens parameters
            if self.parameters_to_fit[i] == 't_0':
                index = 0
            elif self.parameters_to_fit[i] == 'u_0':
                index = 1
            elif self.parameters_to_fit[i] == 't_E':
                index = 2
            elif self.parameters_to_fit[i] == 'rho':
                index = 3
            elif self.parameters_to_fit[i] == 'pi_E_N':
                index = 7
            elif self.parameters_to_fit[i] == 'pi_E_E':
                index = 8
            else:
                raise IndexError('i > n_params')
        else:
            # flux parameters
            # i = n_params + 2. * n + 0; fs
            # i = n_params + 2. * n + 1; fb
            # index = 9 + 3 * n + 0; fs
            # index = 9 + 3 * n + 1 ; fb

            # source flux
            nob = np.where(np.array(self.my_func.fs_indices) == i)
            if len(nob[0]) > 0:
                if len(nob[0]) == 1:
                    index = int(9 + 3 * nob[0][0])
                else:
                    raise AttributeError('Multiple matches in fs_indices.')

            else:
                # blend flux
                nob = np.where(np.array(self.my_func.fb_indices) == i)
                if len(nob[0]) == 1:
                    index = int(9 + 3 * nob[0][0] + 1)
                else:
                    raise AttributeError('Multiple matches in fb_indices.')

        return index

    def _t0_correction(self, value):
        if value > 12000.:
            return value - 2450000.
        else:
            return value

    def _compare_vector(
            self, my_vector, sfit_vector, decimal=5, verbose=False):
        for i, value0 in enumerate(my_vector):
            index = self._get_index(i)

            if i == 0:
                value = self._t0_correction(value0)
            else:
                value = value0

            if verbose:
                print(i, index)
                print(value, sfit_vector[index])

            if value != 0.0:
                np.testing.assert_almost_equal(
                    value / sfit_vector[index], 1., decimal=decimal)
            else:
                np.testing.assert_almost_equal(
                    value,  sfit_vector[index], decimal=decimal)

    def _compare_matrix(self, my_matrix, sfit_matrix, verbose=False, decimal=5):
        n_elements = my_matrix.shape[0]
        for i in range(n_elements):
            ind_i = self._get_index(i)
            for j in range(n_elements):
                ind_j = self._get_index(j)

                if verbose:
                    print(i, j, ind_i, ind_j)
                    print(my_matrix[i, j], sfit_matrix[ind_i, ind_j])

                if my_matrix[i, j] != 0.0:
                    np.testing.assert_almost_equal(
                        my_matrix[i, j] / sfit_matrix[ind_i, ind_j], 1.,
                        decimal=decimal)
                else:
                    np.testing.assert_almost_equal(
                        my_matrix[i, j], sfit_matrix[ind_i, ind_j],
                        decimal=decimal)


def test_cmat():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    comparison_dir = 'PSPL_1_{0}'.format(0.1)

    test = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit)
    test.my_func.update_all(theta0=test.initial_guess)

    n_elements = int(np.sqrt(len(test.matrices[0].c)))
    shape = (n_elements, n_elements)
    sfit_cmat = test.matrices[0].c.reshape(shape)

    for i in range(test.n_params):
        ind_i = test._get_index(i)
        for j in range(test.n_params):
            ind_j = test._get_index(j)
            my_el = test.my_func.cmat[i, j]
            sfit_el = sfit_cmat[ind_i, ind_j]
            if np.abs(sfit_el) > 1.:
                np.testing.assert_almost_equal(my_el / sfit_el, 1., decimal=6)
            else:
                np.testing.assert_almost_equal(my_el, sfit_el, decimal=6)


def test_pspl_1():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    for fac in [0.1, 0.01]:
        comparison_dir = 'PSPL_1_{0}'.format(fac)
        print(comparison_dir)
        test = ComparisonTest(
            datafiles=datafiles, comp_dir=comparison_dir,
            parameters_to_fit=parameters_to_fit)
        test.run()


def test_pspl_2():
    datafiles = ['PSPL_2_Obs_1.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    for fac in [0.1, 0.01]:
        comparison_dir = 'PSPL_2_{0}'.format(fac)
        print(comparison_dir)
        test = ComparisonTest(
            datafiles=datafiles, comp_dir=comparison_dir,
            parameters_to_fit=parameters_to_fit)
        test.run()


def test_pspl_par():
    datafiles = ['PSPL_par_Obs_1.pho', 'PSPL_par_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E', 'pi_E_N', 'pi_E_E']
    coords = "18:00:00 -30:00:00"
    for fac in [0.01]:
        comparison_dir = 'PSPL_par_{0}'.format(fac)
        print(comparison_dir)
        test = ComparisonTest(
            datafiles=datafiles, comp_dir=comparison_dir,
            parameters_to_fit=parameters_to_fit, coords=coords,
            verbose=False)
        test.test_final_results()


def test_pspl_fbzero():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    fac = 0.01
    comparison_dir = 'PSPL_1_{0}_fbzero'.format(fac)
    print(comparison_dir)
    test = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[0., False],
        verbose=True)
    test.run()

def test_pspl_fs_fixed():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    fac = 0.01
    comparison_dir = 'PSPL_1_{0}_fs_fixed'.format(fac)
    print(comparison_dir)
    test = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[False, 2.1],
        verbose=True)
    test.run()

def test_pspl_Obs1_fixed():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    fac = 0.01
    comparison_dir = 'PSPL_1_{0}_Obs1FixedFluxes'.format(fac)
    print(comparison_dir)
    test = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[1.3, False],
        fix_blend_flux=[0.0, False],
        verbose=True)
    test.run()

def test_flux_indexing():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    fac = 0.01
    comparison_dir = 'PSPL_1_{0}_fbzero'.format(fac)
    print(comparison_dir)

    test_1 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[0., False],
        verbose=True)

    assert test_1.my_func.fs_indices == [3, 4]
    assert test_1.my_func.fb_indices == [None, 5]

    test_2 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[False, 0.],
        verbose=True)

    assert test_2.my_func.fs_indices == [3, 5]
    assert test_2.my_func.fb_indices == [4, None]

    test_3 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[0., 0.],
        verbose=True)

    assert test_3.my_func.fs_indices == [3, 4]
    assert test_3.my_func.fb_indices == [None, None]

    test_4 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[False, False],
        verbose=True)

    assert test_4.my_func.fs_indices == [3, 5]
    assert test_4.my_func.fb_indices == [4, 6]

    test_5 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[False, 2.1],
        verbose=True)

    assert test_5.my_func.fs_indices == [3, None]
    assert test_5.my_func.fb_indices == [4, 5]

    test_6 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[0.0, False],
        verbose=True)

    print(test_6.my_func.fs_indices)
    print(test_6.my_func.fb_indices)
    assert test_6.my_func.fs_indices == [None, 4]
    assert test_6.my_func.fb_indices == [3, 5]
