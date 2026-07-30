import time

import numpy as np

from sfit_minimizer.sfit_classes import SFitResults


def set_initial_step_size(options):
    """
    Set the step size for the :py:func:`~minimize` function.

    Arguments:
        options: *dict*
            To specify the step, set options['step'] = *float*. The parameter
            values will be iterated by this value * the calculated step size.
            Alternatively, options['step'] = 'adaptive' will start with a
            default size of 0.01. If None or 'step' not in options, uses a
            default step size of 0.1.

    Returns:
        fac: *float*
            The initial fraction of the step that will be added in each
            iteration of the :py:func:`~minimize` routine.
    """
    default = 0.1
    if (options is None) or not ('step' in options.keys()):
        options = {'step': default}

    if isinstance(options['step'], (str)):
        if options['step'].lower() == 'adaptive':
            fac = 0.01
        else:
            raise ValueError(
                'Invalid option for "step": {0}'.format(options['step']))
    elif options['step'] is None:
        fac = default
    elif np.isfinite(options['step']):
        fac = options['step']
    else:
        raise ValueError(
            'Invalid option for "step": {0}'.format(options['step']))

    return (fac, options)


def minimize(
        sfit_obj, x0=None, tol=1e-3, options=None, max_iter=1000,
        verbose=False):

    """
    Find the best-fit parameters for a function f using A. Gould's sfit
    algorithm.

    Arguments:
        sfit_obj: :py:class:`sfit_minimizer.sfit_classes.SFitFunction`
            The function whose parameters are being fit.

    Keywords:
        x0: *list*, or list-like
            Initial guess for the values of the parameters

        tol: *float*
            Chi2 tolerance; i.e., when chi2_old - chi2_new < tol, stop.

        options: *dict*
            options for the minimizer. Currently available: 'step' (see
            :py:func:`~set_initial_step_size()`)

        max_iter: *int*
            Maximum number of iterations.

        verbose: *bool*
            True = print results of each iteration to the screen.

    Returns:
        :py:class:`sfit_minimizer.sfit_classes.SFitResults` object.

    """

    (fac, options) = set_initial_step_size(options)

    sfit_obj.update_all(x0, verbose=verbose)
    old_chi2 = sfit_obj.chi2
    x_old = x0
    if verbose:
        print('{6} {0:>16} {1:>16} {2} {3}\n{4}\n{5}\n'.format(
                'old_chi2', 'new_chi2', '[step]', 'stepfrac', '[old params]',
                '[new params]', 'i'))
        print('{6} {0:16.4f} {1:16.4f} {2} {3}\n{4}\n{5}\n'.format(
                old_chi2, sfit_obj.chi2, sfit_obj.step, fac, x_old, None, -1))
        start = time.time()

    for i in range(max_iter):
        # print('minimize steps', x_old, fac, sfit_obj.step)
        x_new = x_old + fac * sfit_obj.step
        sfit_obj.update_all(x_new)

        if verbose:
            print('{6} {0:16.4f} {1:16.4f} {2} {3}\n{4}\n{5}\n'.format(
                old_chi2, sfit_obj.chi2, sfit_obj.step, fac, x_old, x_new, i))
            stop = time.time()
            print('step runtime (s): ', stop - start)
            start = time.time()

        if options['step'] == 'adaptive':
            if old_chi2 - sfit_obj.chi2 < 1.0:
                fac = 0.1

        if old_chi2 < sfit_obj.chi2:
            msg = 'New chi2 worse than old chi2.\n'
            msg += 'Previous step: {0}, {1}\n'.format(old_chi2, x_old)
            msg += 'New step: {0}, {1}\n'.format(sfit_obj.chi2, x_new)
            sfit_obj.update_all(x_old)
            return SFitResults(
                sfit_obj, success=False, msg=msg, iterations=i)
        elif old_chi2 - sfit_obj.chi2 < tol:
            if verbose:
                print('tolerance reached!')

            break
        else:
            old_chi2 = sfit_obj.chi2
            x_old = x_new

    sfit_obj.update_all(x_new)
    if i < max_iter - 1:
        return SFitResults(
            sfit_obj, success=True, iterations=i, fun=sfit_obj.chi2)
    else:
        return SFitResults(
            sfit_obj, success=False, fun=old_chi2,
            msg='max iterations exceeded: {0}'.format(max_iter), iterations=i)
