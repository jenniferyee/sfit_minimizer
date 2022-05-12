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
                 coords=None, n_t_star=None, verbose=False, fix_blend_flux=None,
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
            if filename[0:2] == 'FS':
                filestr = filename.split('.')
                bandpass = filestr[0][-1]
            else:
                bandpass = None

            data = mm.MulensData(
                file_name=os.path.join(data_path, filename), phot_fmt='mag',
                bandpass=bandpass)
            self.datasets.append(data)

            if (9 + i * 3) >= len(self.matrices[0].a):
                flux_guess = [1.0, 0.0]
            else:
                flux_guess = [self.matrices[0].a[9 + i * 3],
                              self.matrices[0].a[9 + i * 3 + 1]]

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

        gammas = {'I': self.matrices[0].a[4],
                  'V': self.matrices[0].a[5],
                  'H': self.matrices[0].a[6]}
        if 'rho' in self.parameters_to_fit:
            if n_t_star is None:
                n_t_star = 10.

            t_star = self.model.parameters.rho * self.model.parameters.t_E
            self.model.set_magnification_methods([
                self.model.parameters.t_0 - n_t_star * t_star,
                'finite_source_LD_Yoo04_direct',
                self.model.parameters.t_0 + n_t_star * t_star])
            for band, value in gammas.items():
                self.model.set_limb_coeff_gamma(band, value)

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
            np.testing.assert_allclose(self.my_func.step[i], 0.0, atol=0.001)

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
            self.my_func.dvec, sfit_matrix.d, decimal=3, verbose=self.verbose)

        # c matrix
        cmat = sfit_matrix.c.reshape(shape)
        self._compare_matrix(
            self.my_func.cmat, cmat, decimal=2, verbose=self.verbose)

        # step
        self._compare_vector(
            self.my_func.step, sfit_matrix.da, decimal=3, verbose=self.verbose)

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
            dataset_chi2 = self.my_func.event.get_chi2_for_dataset(i)
            if dataset_chi2 < 10.:
                np.testing.assert_allclose(
                    np.sum(sfit_matrix.chi2[i]), dataset_chi2, atol=0.1)
            else:
                np.testing.assert_allclose(
                    np.sum(sfit_matrix.chi2[i]), dataset_chi2, rtol=0.001)

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
        """ if my_vector is 0., uses an absolute tolerance of 10.^decimal.
        Otherwise, uses a relative tolerance."""
        if verbose:
            for i, value0 in enumerate(my_vector):
                index = self._get_index(i)
                print(i, index)
                print(value0, sfit_vector[index], value0 / sfit_vector[index])

        rtol = 10. ** (-decimal)
        if 'rho' in self.parameters_to_fit:
            rtol = 0.05
            
        for i, value0 in enumerate(my_vector):
            index = self._get_index(i)

            if i == 0:
                value = self._t0_correction(value0)
            else:
                value = value0

            if np.abs(value) > 10.e-6:
                np.testing.assert_allclose(
                    value, sfit_vector[index], rtol=rtol)
            else:
                np.testing.assert_allclose(
                    value, sfit_vector[index], atol=10.**(-decimal))

    def _compare_matrix(self, my_matrix, sfit_matrix, verbose=False, decimal=5):

        if verbose:
            print('parameters', self.parameters_to_fit)
        n_elements = my_matrix.shape[0]
        if verbose:
            for i in range(n_elements):
                ind_i = self._get_index(i)
                for j in range(n_elements):
                    # for j in [i]:  # testing code
                    ind_j = self._get_index(j)
                    print(i, j, ind_i, ind_j)
                    print(
                        my_matrix[i, j], sfit_matrix[ind_i, ind_j],
                        my_matrix[i, j] / sfit_matrix[ind_i, ind_j])

        for i in range(n_elements):
            ind_i = self._get_index(i)
            for j in range(n_elements):
                ind_j = self._get_index(j)

                rtol = 10.**(-decimal)
                if 'rho' in self.parameters_to_fit:
                    if i < len(self.parameters_to_fit):
                        if self.parameters_to_fit[i] == 'rho':
                            rtol = 0.05

                    if j < len(self.parameters_to_fit):
                        if self.parameters_to_fit[j] == 'rho':
                            rtol = 0.05

                if np.abs(my_matrix[i, j]) > 10e-6:
                    np.testing.assert_allclose(
                        my_matrix[i, j], sfit_matrix[ind_i, ind_j],
                        rtol=rtol)
                else:
                    np.testing.assert_allclose(
                        my_matrix[i, j], sfit_matrix[ind_i, ind_j],
                        atol=10.**(-decimal))


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
        verbose=False)
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
        verbose=False
    )
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
        verbose=False)
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
        verbose=False)

    assert test_1.my_func.fs_indices == [3, 4]
    assert test_1.my_func.fb_indices == [None, 5]

    test_2 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[False, 0.],
        verbose=False)

    assert test_2.my_func.fs_indices == [3, 5]
    assert test_2.my_func.fb_indices == [4, None]

    test_3 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[0., 0.],
        verbose=False)

    assert test_3.my_func.fs_indices == [3, 4]
    assert test_3.my_func.fb_indices == [None, None]

    test_4 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_blend_flux=[False, False],
        verbose=False)

    assert test_4.my_func.fs_indices == [3, 5]
    assert test_4.my_func.fb_indices == [4, 6]

    test_5 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[False, 2.1],
        verbose=False)

    assert test_5.my_func.fs_indices == [3, None]
    assert test_5.my_func.fb_indices == [4, 5]

    test_6 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, fix_source_flux=[0.0, False],
        verbose=False)

    print(test_6.my_func.fs_indices)
    print(test_6.my_func.fb_indices)
    assert test_6.my_func.fs_indices == [None, 4]
    assert test_6.my_func.fb_indices == [3, 5]


def test_flux_indexing_2():
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho', 'PSPL_2_Obs_1.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    fac = 0.01
    comparison_dir = 'PSPL_1_{0}_fbzero'.format(fac)
    print(comparison_dir)

    test_7 = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit,
        fix_source_flux=[False, 1.0, False],
        fix_blend_flux=[False, 0.0, False], verbose=False)
    print(test_7.my_func.fs_indices)
    print(test_7.my_func.fb_indices)
    assert test_7.my_func.fs_indices == [3, None, 5]
    assert test_7.my_func.fb_indices == [4, None, 6]


def test_fspl_1():
    """ Test that the FSPL gradient calculation is very accurate."""
    datafiles = ['FSPL_Obs_1_I.pho', 'FSPL_Obs_2_V.pho']
    parameters_to_fit = ['t_0', 'u_0', 't_E', 'rho']
    fac = 0.01
    comparison_dir = 'FSPL_1_{0}'.format(fac)
    print(comparison_dir)
    test = ComparisonTest(
        datafiles=datafiles, comp_dir=comparison_dir,
        parameters_to_fit=parameters_to_fit, n_t_star=100,
        verbose=True)
    test.test_3_iterations()


def test_fixed_fluxes():
    """Check that the calc_df works for a variety of cases."""
    # datasets
    datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho', 'PSPL_2_Obs_1.pho']
    datasets = []
    n_data = []
    for i, filename in enumerate(datafiles):
        data = mm.MulensData(
            file_name=os.path.join(data_path, filename), phot_fmt='mag')
        n_data.append(len(data.time))
        datasets.append(data)

    print('MM.version', mm.__version__)
    print('datasets', datasets)
    print('n_data', n_data)

    # model parameters
    model_params = {'t_0': 8645.00000, 'u_0': 0.250000, 't_E': 25.2000}

    def run_test(ulens=None, fix_source_flux=None, fix_blend_flux=None):
        # Create PSPL Function object
        if ulens:
            parameters_to_fit = ['t_0', 'u_0', 't_E']
            n_ulens = 3
            initial_guess = [model_params[key] for key in parameters_to_fit]
        else:
            parameters_to_fit = []
            n_ulens = 0
            initial_guess = []

        for i in range(len(datasets)):
            if i != fix_source_flux:
                initial_guess.append(1.0)

            if i != fix_blend_flux:
                initial_guess.append(0.0)

        n_params = n_ulens + 2 * len(datasets)
        if fix_source_flux is not None:
            fix_source_flux_dict = {datasets[fix_source_flux]: 1.0}
            n_params -= 1
        else:
            fix_source_flux_dict = None

        if fix_blend_flux is not None:
            fix_blend_flux_dict = {datasets[fix_blend_flux]: 0.0}
            n_params -= 1
        else:
            fix_blend_flux_dict = None

        print('initial guess', initial_guess)
        event = mm.Event(
            model=mm.Model(model_params), datasets=datasets,
            fix_source_flux=fix_source_flux_dict,
            fix_blend_flux=fix_blend_flux_dict)
        print('fix_source_flux', event.fix_source_flux)
        print('fix_blend_flux', event.fix_blend_flux)

        my_func = sfit_minimizer.mm_funcs.PSPLFunction(
            event, parameters_to_fit)
        print('fs_indices', my_func.fs_indices)
        print('fb_indices', my_func.fb_indices)
        my_func._update_ulens_params(initial_guess)

        # Run calc_df
        my_func.calc_df()

        # check the shape of df
        assert my_func.df.shape == (n_params, np.sum(n_data))

        # check that things that should be zero are zero
        for i in range(len(datasets)):
            if i == 0:
                ind_i_0 = 0
            else:
                ind_i_0 = np.sum(n_data[0:i]).astype(int)

            if i == (len(datasets) - 1):
                ind_i_1 = np.sum(n_data).astype(int)
            else:
                ind_i_1 = np.sum(n_data[0:i + 1]).astype(int)

            for j in range(len(datasets)):
                print('i,j', i, j)
                if j != i:
                    # zeros for dataset i for flux parameters of dataset j
                    if fix_source_flux != j:
                        assert (
                            np.sum(
                                my_func.df[
                                    my_func.fs_indices[j], ind_i_0:ind_i_1]
                            ) == 0)

                    if fix_blend_flux != j:
                        assert (
                            np.sum(
                                my_func.df[
                                    my_func.fb_indices[j], ind_i_0:ind_i_1]
                            ) == 0)

                else:
                    # zeros for dataset i for flux parameters of dataset j
                    if fix_source_flux != j:
                        assert (
                            np.sum(
                                my_func.df[
                                    my_func.fs_indices[j], ind_i_0:ind_i_1]
                            ) != 0)

                    if fix_blend_flux != j:
                        assert (
                            np.sum(
                                my_func.df[
                                    my_func.fb_indices[j], ind_i_0:ind_i_1]
                            ) != 0)

    # ulens parameters + 3 datasets
    print('t1')
    run_test(ulens=True)

    # ulens parameters + fix source flux for dataset 1
    print('t2')
    run_test(ulens=True, fix_source_flux=0)

    # ulens parameters + fix blend flux for dataset_1
    print('t3')
    run_test(ulens=True, fix_blend_flux=0)

    # ulens parameters + fix both source and blend flux for dataset_1
    print('t4')
    run_test(ulens=True, fix_source_flux=0, fix_blend_flux=0)

    # ulens parameters + fix both source and blend flux for dataset_2
    print('t5')
    run_test(ulens=True, fix_source_flux=1, fix_blend_flux=1)

    # ulens parameters + fix both source and blend flux for dataset_3
    print('t6')
    run_test(ulens=True, fix_source_flux=2, fix_blend_flux=2)

    # just fluxes
    print('t7')
    run_test(ulens=False)

    # just fluxes but fix source flux for dataset_1
    print('t8')
    run_test(ulens=False, fix_source_flux=0)

    # just fluxes but fix blend flux for dataset_1
    print('t9')
    run_test(ulens=False, fix_blend_flux=0)

    # just fluxes but fix both source and blend flux for dataset_1
    print('t10')
    run_test(ulens=False, fix_source_flux=0, fix_blend_flux=0)

    # just fluxes but fix both source and blend flux for dataset_3
    print('t11')
    run_test(ulens=False, fix_source_flux=1, fix_blend_flux=1)

    # just fluxes but fix both source and blend flux for dataset_3
    print('t12')
    run_test(ulens=False, fix_source_flux=2, fix_blend_flux=2)

    # just fluxes but fix source and blend flux for different datasets
    print('t13')
    run_test(ulens=False, fix_source_flux=2, fix_blend_flux=0)
