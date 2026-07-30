import numpy as np


class SFitResults(object):
    """
    Results of the minimization algorithm. Modeled after
    scipy.optimize.OptimizeResult.

    Arguments:
        func: :py:class:`~SFitFunction`

    Keywords:
        success: *bool*
            see :py:attr:`~success`

        msg: *str*
            see :py:attr:`~msg`

    """

    def __init__(
        self, func, success=None, msg=None, iterations=None, fun=None
    ):
        self._x = func.theta
        self._sigmas = func.get_sigmas()
        self._success = success
        self._msg = msg
        self._iter = iterations
        self._fun = fun

    def __repr__(self):
        output_str = "fun = {0}\n".format(self.fun)
        output_str += "sigmas = {0}\n".format(self.sigmas)
        output_str += "message = {0}\n".format(self.msg)
        output_str += "nit = {0}\n".format(self.nit)
        output_str += "success = {0}\n".format(self.success)
        output_str += "x = {0}\n".format(self.x)

        return output_str

    @property
    def x(self):
        """Best-fit (last calculated) value of the parameters."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def fun(self):
        """final function evaluation, e.g., chi2"""
        return self._fun

    @fun.setter
    def fun(self, value):
        self._fun = value

    @property
    def sigmas(self):
        """Uncertainties for :py:attr:`~x`"""
        return self._sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._sigmas = value

    @property
    def success(self):
        """True = algorithm completed successfully. False = algorithm ran into
        a limit or failure mode."""
        return self._success

    @success.setter
    def success(self, value):
        self._success = value

    @property
    def msg(self):
        """If :py:attr:`~success` == False, some information about the failure
        mode."""
        return self._msg

    @msg.setter
    def msg(self, value):
        self._msg = value

    @property
    def nit(self):
        """Total number of iterations executed."""
        return self._iter

    @nit.setter
    def nit(self, value):
        self._iter = value


class SFitFunction(object):
    """
    Main class for sfit minimization routine. Establishes all the necessary
    functions for executing the A. Gould algorithm. At a minimum, the user
    should specify:

      EITHER :py:func:`~calc_model` OR :py:func:`~calc_residuals`

      AND :py:func:`~calc_df`.

    Arguments:
        data: *np.array* with shape (N, 3)
            an array of the data to be fitted with columns (x, y, yerr).

        theta: *list* of length M
            trial values for the parameters of the function to be fit.

    """

    def __init__(self, data=None, theta=None):
        if isinstance(data, np.ndarray):
            if data.shape[1] == 3:
                self.data = data
            else:
                raise ValueError(
                    "data must have shape (N, 3). Shape: {0}".format(
                        data.shape
                    )
                )

        else:
            raise TypeError(
                "data must be an np.array object with shape (N, 3). "
                + "Type: {0}".format(type(data))
            )

        self.theta = theta
        self.reset_all()

    def reset_all(self):
        self._ymod = None
        self._residuals = None
        self._chi2 = None
        self._df = None
        self._dchi2 = None
        self._dvec = None
        self._bmat = None
        self._cmat = None
        self._sigmas = None
        self._step = None

    # theta
    @property
    def theta(self):
        """
        *list* or *np.array*

        Parameters of the function being fit to the data. Setting this
        parameter will reset all other parameters to None. Use
        :py:func:`~update_all()` to update everything.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        self.reset_all()
        if value is None:
            self._theta = value
        else:
            if isinstance(value, (list, np.ndarray)):
                self._theta = value
            else:
                raise TypeError(
                    "theta must be either list or np.array. "
                    + "Type: {0}".format(type(value))
                )

    def update_all(self, theta=None, verbose=False):
        """
        Recalculate all of the data properties with respect to the new model
        parameters.

        Keywords:
            theta0: *list* of length M, optional
                new trial values for the parameters of the function to be fit.
                If not provided, recalculate using the current value of theta.

            verbose: *bool*, optional
                Default is False. If True, prints output after each stage for
                debugging.

        """
        if theta is not None:
            self.reset_all()
            self.theta = theta

        self.calc_residuals()
        if verbose:
            print("residuals", self.residuals)
            print(
                "res min/max",
                np.min(self.residuals),
                np.max(self.residuals),
                np.max(np.abs(self.residuals)),
            )

        self.calc_chi2()
        if verbose:
            print("chi2", self.chi2)

        self.calc_df()
        if verbose:
            print("df", self.df)

        self.calc_dchi2()
        if verbose:
            print("dchi2", self.dchi2)

        self.calc_dvec()
        if verbose:
            print("dvec", self.dvec)

        self.calc_bmat()
        if verbose:
            print("bmat", self.bmat)
            print(
                "bmat finite vs. shape:",
                np.sum(np.isfinite((self.bmat))),
                self.bmat.shape,
            )

        self.calc_cmat()
        if verbose:
            print("cmat", self.cmat)

        self.calc_step()
        if verbose:
            print("step", self.step)

    # Model Values
    @property
    def ymod(self):
        """
        *np.array* of shape (N), where each element k is the value of the model
        evaluated at data[k, 0].
        """
        return self._ymod

    @ymod.setter
    def ymod(self, value):
        if isinstance(value, np.ndarray):
            if len(value) == len(self.data):
                self._ymod = value
            else:
                raise ValueError(
                    "ymod should have the same length as data. data.shape: "
                    + "{0}, ymod.shape: {1}".format(
                        self.data.shape, value.shape
                    )
                )

        else:
            raise TypeError(
                "ymod must be an np.array object. Type: {0}".format(
                    type(value)
                )
            )

    def calc_model(self):
        """
        FUNCTION SPECIFIC: Calculate expected values of the model. May
        be explicitly defined for a specific class that inherits
        SFitFunction. Only used to calculate :py:attr:`~ymod`, so
        :py:func:`~calc_residuals()` may be defined instead.

        sets :py:attr:`~ymod`

        """
        raise NotImplementedError(
            "User must define calc_model() for a child class of SFitFunction."
        )

    # Residuals
    @property
    def residuals(self):
        """
        *np.array* of shape (N), where each element
        k = :py:attr: data[k, 1] - `~ymod` [k]
        """
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        if isinstance(value, np.ndarray):
            if len(value) == len(self.data):
                self._residuals = value
            else:
                raise ValueError(
                    "residuals should have the same length as data. "
                    + "data.shape: {0}, residuals.shape: {1}".format(
                        self.data.shape, value.shape
                    )
                )
        else:
            raise TypeError(
                "residuals must be an np.array object. "
                + "Type: {0}".format(type(value))
            )

    def calc_residuals(self):
        """
        Difference between the data and the model. Either this OR
        :py:func:`~calc_model()` should be explicitly defined for a specific
        class that inherits SFitFunction.

        sets :py:attr:`~residuals`
        """
        if self.ymod is None:
            self.calc_model()

        self.residuals = self.data[:, 1] - self.ymod

    # Chi2
    @property
    def chi2(self):
        """
        *float*

        chi2 of the model with respect to the data.
        chi2 = Sum_k (residuals[k]^2) / data[k, 2]^2
        """
        return self._chi2

    def calc_chi2(self):
        """
        Calculate the chi2 of the data relative to the model.

        sets :py:attr:`~chi2`
        """
        if self.residuals is None:
            self.calc_residuals()

        self._chi2 = np.sum(self.residuals**2 / self.data[:, 2] ** 2)

    def get_chi2(self):
        """Calculate and return the chi2 of the data w.r.t. the model. See
        :py:func:`~calc_chi2`."""
        self.calc_chi2()

        return self.chi2

    # df
    @property
    def df(self):
        """
        *np.array* of shape (M, N), where df[i, k] returns the partial
        derivative of the *fitting function* with respect to parameter i
        evaluated at point k. shape = (len(theta), len(data))
        """
        if self._df is None:
            self.calc_df()

        return self._df

    @df.setter
    def df(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == (len(self.theta), len(self.data)):
                self._df = value
            else:
                raise ValueError(
                    "df should have the shape (M, N) where M is the number of"
                    + "parameters and N is the number of data"
                    + " points. M=len(theta): "
                    + "{2}, N=data.shape: {0}, df.shape: {1}".format(
                        self.data.shape, value.shape, len(self.theta)
                    )
                )
        else:
            raise TypeError(
                "df must be an np.array object. Type: {0}".format(type(value))
            )

    def calc_df(self):
        """
        FUNCTION SPECIFIC: Calculate the derivatives of the fitting
        function relative to each parameter theta and store as
        self.df.  Should be explicitly defined for a specific class
        that inherits SFitFunction.

        sets :py:attr:`~df`
        """
        raise NotImplementedError(
            "User must define calc_df() for a child class of SFitFunction."
        )

    # Sigmas
    @property
    def sigmas(self):
        """
        *np.array* of length M

        Uncertainty in each parameter theta: sigma_theta_i = sqrt(c_ii)"""
        return self._sigmas

    def calc_sigmas(self):
        """
        Calculate the uncertainty in each parameter :py:attr:`~theta`.
        """
        if self.cmat is None:
            self.calc_cmat()

        self._sigmas = np.sqrt(np.diagonal(self.cmat))

    def get_sigmas(self):
        """Calculate and return the uncertainty in each parameter theta. see
        :py:func:`~calc_sigmas`"""
        self.calc_sigmas()

        return self.sigmas

    # Step
    @property
    def step(self):
        """
        *np.array* of shape (M)

        Full proposed step size for each parameter theta_i:
            theta_new_i = theta_old_i + step_i where
            step_i = sum_j c_ij * d_j.
        """
        return self._step

    def calc_step(self):
        """
        Calculate :py:attr:`~step`.
        """
        if self.cmat is None:
            self.calc_cmat()

        if self.dvec is None:
            self.calc_dvec()

        self._step = np.sum(self.cmat * self.dvec, axis=1)

    def get_step(self):
        """Calculate and return the full step for each parameter theta_i. See
        :py:func:`~calc_step`."""
        self.calc_step()

        return self.step

    # dchi2 (chi2 gradient)
    @property
    def dchi2(self):
        """
        *np.array* of shape (M, N)

        Partial derivatives of the *chi2* with respect
        to the parameters, calculated at each data point:

        dchi2_i,k = -2 * :py:attr:`~residuals` * :py:attr:`~df` / data[k, 2]**2

        shape = (len(theta), len(data))
        """
        return self._dchi2

    def calc_dchi2(self):
        """Calculate the gradient of chi2 (:py:attr:`~dchi2`) at each
        datapoint."""

        if self.df is None:
            self.calc_df()

        if self.residuals is None:
            self.calc_residuals()

        chi2_gradient = []
        for i in range(len(self.theta)):
            chi2_gradient.append(
                -2 * self.residuals * self.df[i] / self.data[:, 2] ** 2
            )

        self._dchi2 = np.array(chi2_gradient)

    def get_dchi2(self):
        """Calculate and return the partial derivatives of the fitting
        function. See :py:func:`~calc_dchi2`."""
        self.calc_df()

        return np.sum(self.df, axis=1)

    def get_partials(self):
        """Alternative to :py:func:`~get_dchi2`."""
        return self.get_dchi2()

    # d vector
    @property
    def dvec(self):
        """
        *np.array* of shape (M)

        d vector from sfit, used to calculate the :py:attr:`~step` size:
            d_i = - Sum_k (partial chi2 / partial theta_i) / 2

        shape = ( len(theta) )
        """
        return self._dvec

    def calc_dvec(self):
        """
        Calculate the d vector and store it as :py:attr:`~dvec`.
        """
        if self.dchi2 is None:
            self.calc_dchi2()

        self._dvec = -np.sum(self.dchi2, axis=1) / 2.0

    def get_dvec(self):
        """Calculate and return the d vector. See :py:func:`~calc_dvec`."""
        self.calc_dvec()

        return self.dvec

    # b matrix
    @property
    def bmat(self):
        """
        *np.array* of shape (M, M)

        b matrix from sfit. invert to find c matrix (:py:attr:`~cmat`) used for
        stepping and finding sigmas:'

            b_ij
            = Sum_k (
            (partial F / partial theta_i)(partial F/partial theta_j) /
            data[k, 2]^2 )

        where F is the *fitting function* and data[k, 2] are the errors.
        shape = ( len(theta), len(theta) ).
        """
        return self._bmat

    def calc_bmat(self):
        """
        Calculate the b matrix (:py:attr:`~bmat`).
        """
        if self.df is None:
            self.calc_df()

        self._bmat = np.inner(self.df, self.df / self.data[:, 2] ** 2)

    def get_bmat(self):
        """Calculate and return the b matrix. See :py:func:`~calc_bmat`."""
        self.calc_bmat()

        return self.bmat

    # c matrix
    @property
    def cmat(self):
        """
        *np.array* of shape (M, M)

        c matrix from sfit:
            c = inv(:py:attr:`~bmat`)

        shape = ( len(theta), len(theta) )"""
        return self._cmat

    def calc_cmat(self):
        """Invert the b matrix (:py:attr:`~bmat`) to find the c matrix
        (:py:attr:`~cmat`)."""
        if self.bmat is None:
            self.calc_bmat()

        self._cmat = np.linalg.inv(self.bmat)

    def get_cmat(self):
        """Calculate and return the c matrix. See :py:func:`~calc_cmat`."""
        self.calc_cmat()

        return self.cmat
