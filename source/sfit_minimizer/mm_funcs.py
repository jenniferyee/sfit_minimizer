import numpy as np
import matplotlib.pyplot as plt
import sfit_minimizer


def fit_mulens_event(
        event, parameters_to_fit=None, initial_guess=None, plot=False,
        verbose=False):
    # Setup the fitting
    if parameters_to_fit is None:
        parameters_to_fit = event.model.parameters.parameters.keys()

    if initial_guess is None:
        initial_guess = []
        for key in parameters_to_fit:
            if key == 't_E':
                initial_guess.append(event.model.parameters.parameters[key].value)
            else:
                initial_guess.append(event.model.parameters.parameters[key])

        event.fit_fluxes()
        source_fluxes = event.source_fluxes
        blend_fluxes = event.blend_fluxes
        for i in range(len(event.datasets)):
            initial_guess.append(source_fluxes[i][0])
            initial_guess.append(blend_fluxes[i])

    my_func = sfit_minimizer.mm_funcs.PointLensSFitFunction(
        event, parameters_to_fit)

    if verbose:
        print(event)
        print('fixed fs', event.fix_source_flux)
        print('fixed fb', event.fix_blend_flux)
        print('Initial guess', initial_guess)

    # Do the fit
    result = sfit_minimizer.minimize(
        my_func, x0=initial_guess, tol=1e-5,
        options={'step': 'adaptive'}, verbose=verbose)

    values = result.x
    sigmas = result.sigmas
    if verbose:
        # Print the results
        print('Full Results:')
        print(result)
        print('results: ')
        print(values)
        print('+/-')
        print(sigmas)

    my_func.update_all(values)
    if verbose:
        print('chi2: ', my_func.chi2)

    if plot:
        if my_func.event.model.parameters.t_0 > 2000000:
            subtract_2450000 = True
        else:
            subtract_2450000 = False

        # Plot results
        my_func.event.plot(subtract_2450000=subtract_2450000, show_bad=True)
        plt.show()

    return result


class PointLensSFitFunction(sfit_minimizer.SFitFunction):
    """
    A class for fitting a point-source point-lens microlensing light curve to
    observed data using :py:func:`sfit_minimizer.sfit_minimize.minimize()`.
    Simultaneously fits microlensing parameters and source and blend fluxes for
    each dataset.

    Arguments:
        event: *MulensModel.Event()* object
            event contains datasets and an initial model.

        parameters_to_fit: *list* of *str*
            list of the named model parameters to be fit. (Not including the
            fluxes.)

    Notes:
        1. if you want to fix the source or blend flux for a particular dataset,
    use the *fix_source_flux* or *fix_blend_flux* keywords in *event* as usual.

        2. If u_0 is a parameter of the fit and it is too close to zero,
    the matrix inversion will fail (produce
    numpy.linalg.LinAlgError: Singular matrix). So if you see this error, check
    the value of u_0. (Probably true of other parameters as well)

    """

    def __init__(
            self, event, parameters_to_fit, estimate_fluxes=False,
            add_2450000=False):
        self.event = event
        self.parameters_to_fit = parameters_to_fit

        self.fs_indices = None
        self.fb_indices = None
        self.n_params = None
        self.set_flux_indices()
        self._initialize_fluxes(estimate_fluxes)

        self.data_len = None
        self.flatten_data()
        self._add_2450000 = add_2450000

    @property
    def add_2450000(self):
        """
        *bool*

        if True: add 2450000 to theta[t_0] before putting it in the model.
        """
        return self._add_2450000

    def flatten_data(self):
        flattened_data = self._flatten_data()
        sfit_minimizer.SFitFunction.__init__(self, data=flattened_data)

    def set_flux_indices(self):
        # must be carried out before initialize_fluxes()
        self.fs_indices = []
        self.fb_indices = []
        n_params = len(self.parameters_to_fit)
        for dataset in self.event.datasets:
            fs_index = n_params
            n_params += 1
            if dataset in self.event.fix_source_flux.keys():
                if self.event.fix_source_flux[dataset] is not False:
                    fs_index = None
                    n_params -= 1

            self.fs_indices.append(fs_index)

            fb_index = n_params
            n_params += 1
            if dataset in self.event.fix_blend_flux.keys():
                if self.event.fix_blend_flux[dataset] is not False:
                    fb_index = None
                    n_params -= 1

            self.fb_indices.append(fb_index)

        self.n_params = n_params

    def _initialize_fluxes(self, estimate_fluxes):
        # Set initial source fluxes
        fix_source_flux = {}
        fix_blend_flux = {}
        if estimate_fluxes:
            self.event.fit_fluxes()
            for dataset in self.event.datasets:
                (source_flux, blend_flux) = self.event.get_flux_for_dataset(
                    dataset)
                fix_source_flux[dataset] = source_flux
                fix_blend_flux[dataset] = blend_flux
        else:
            for dataset in self.event.datasets:
                if dataset in self.event.fix_source_flux.keys():
                    fix_source_flux[dataset] = self.event.fix_source_flux[
                        dataset]
                else:
                    fix_source_flux[dataset] = 1.0

                if dataset in self.event.fix_blend_flux.keys():
                    fix_blend_flux[dataset] = self.event.fix_blend_flux[
                        dataset]
                else:
                    fix_blend_flux[dataset] = 0.0

        self.event.fix_source_flux = fix_source_flux
        self.event.fix_blend_flux = fix_blend_flux

    def _flatten_data(self):
        """ Concatenate good points for all datasets into a single array with
        columns: Date, flux, err.
        """
        self.data_len = []
        flattened_data = []
        for i, dataset in enumerate(self.event.datasets):
            data = [dataset.time[dataset.good], dataset.flux[dataset.good],
                    dataset.err_flux[dataset.good]]
            self.data_len.append(np.sum(dataset.good))
            if i == 0:
                flattened_data = np.array(data)
            else:
                flattened_data = np.hstack((flattened_data, data))

        return flattened_data.transpose()

    def _update_ulens_params(self, theta):
        self._theta = theta

        for (key, val) in enumerate(self.parameters_to_fit):
            if (val == 't_0') and self.add_2450000:
                setattr(
                    self.event.model.parameters, val, theta[key] + 2450000.)
            else:
                setattr(self.event.model.parameters, val, theta[key])

        for i, dataset in enumerate(self.event.datasets):
            if self.fs_indices[i] is not None:
                self.event.fix_source_flux[dataset] = theta[self.fs_indices[i]]

            if self.fb_indices[i] is not None:
                self.event.fix_blend_flux[dataset] = theta[self.fb_indices[i]]

        self.event.fit_fluxes(bad=False)

    def update_all(self, theta=None, verbose=False):
        if theta is None:
            raise ValueError('theta must be passed to update_all()')

        self._update_ulens_params(theta)

        if verbose:
            print('new value:', theta)
            print('fluxes:', self.event.fluxes)
            print('model:', self.event.model)

        sfit_minimizer.SFitFunction.update_all(self, theta, verbose=verbose)

    def calc_residuals(self):
        """Calculate expected values of the residuals"""
        res = []
        for i, fit in enumerate(self.event.fits):
            res_dataset = fit.get_residuals(phot_fmt='flux', bad=False)
            if i == 0:
                res = np.array(res_dataset[0][fit.dataset.good])
            else:
                res = np.hstack(
                    (res, res_dataset[0][fit.dataset.good]))

        self.residuals = res

    def calc_df(self):
        """
        Calculate the derivatives of the fitting function and store as
        self.df.

        """
        data_indices = np.cumsum(self.data_len)
        dfunc = np.zeros((self.n_params, data_indices[-1]))
        for i, fit in enumerate(self.event.fits):
            # Indexes for the ith dataset
            if i == 0:
                ind_start = 0
            else:
                ind_start = data_indices[i - 1]

            ind_stop = data_indices[i]

            # Derivatives of ulens params
            dA_dparm = fit.get_d_A_d_params_for_point_lens_model(
                   self.parameters_to_fit)
            if len(self.parameters_to_fit) > 0:
                for j, key in enumerate(self.parameters_to_fit):
                    x = fit.source_flux * dA_dparm[key]
                    dfunc[j, ind_start:ind_stop] = x

            # Derivatives of flux parameters
            if self.fs_indices[i] is not None:
                dfunc_df_source = np.array(
                    [fit.data_magnification[fit.dataset.good]])
                dfunc[self.fs_indices[i], ind_start:ind_stop] = dfunc_df_source

            if self.fb_indices[i] is not None:
                dfunc_df_blend = np.ones((1, np.sum(fit.dataset.good)))
                dfunc[self.fb_indices[i], ind_start:ind_stop] = dfunc_df_blend

        self.df = dfunc
