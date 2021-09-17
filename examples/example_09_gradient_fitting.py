"""
Fit point source point lens model to OB08092 using Newton-CG method from scipy.
This method requires calculating chi^2 gradient.
"""

import os
import sfit_minimize
import scipy.optimize as op
import matplotlib.pyplot as plt

import MulensModel as mm


def chi2_fun(theta, event, parameters_to_fit):
    """
    for given event set attributes from parameters_to_fit (list of
    str) to values from theta list
    """
    for (key, val) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, val, theta[key])
    return event.get_chi2()


def jacobian(theta, event, parameters_to_fit):
    """
    Calculate chi^2 gradient (also called Jacobian).
    """
    for (key, val) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, val, theta[key])
    return event.chi2_gradient(parameters_to_fit)


# Read in the data file
file_ = os.path.join(mm.DATA_PATH, "photometry_files", "OB08092",
                     "phot_ob08092_O4.dat")
data = mm.MulensData(file_name=file_)

# Initialize the fit
parameters_to_fit = ["t_0", "u_0", "t_E"]
t_0 = 5380.
u_0 = 0.1
t_E = 18.
model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E})

# Link the data and the model
ev = mm.Event(datasets=data, model=model)
print('Initial Trial\n{0}'.format(ev.model.parameters))

# Find the best-fit parameters
initial_guess = [t_0, u_0, t_E]
result = op.minimize(
    chi2_fun, method=sfit_minimize.minimize, x0=initial_guess,
    args=(ev, parameters_to_fit),
    jac=jacobian, tol=1e-3, options={'step': 'adaptive'})
(fit_t_0, fit_u_0, fit_t_E) = result.x

# And their uncertainties
sigmas = result.sigmas

# Save the best-fit parameters
chi2 = chi2_fun(result.x, ev, parameters_to_fit)

# Output the fit parameters
msg = 'Best Fit: t_0 = {0:12.5f}, u_0 = {1:6.4f}, t_E = {2:8.3f}'
print(msg.format(fit_t_0, fit_u_0, fit_t_E))
print('Chi2 = {0:12.2f}'.format(chi2))
print('\nsfit_minimizer.minimize result:')
print(result)

# Plot and compare the two models
init_model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E})
final_model = mm.Model({'t_0': fit_t_0, 'u_0': fit_u_0, 't_E': fit_t_E})
plt.figure()
init_model.plot_lc(data_ref=data, label='Initial Trial')
final_model.plot_lc(data_ref=data, label='Final Model')
plt.title('Difference b/w Input and Fitted Model')
plt.legend(loc='best')

# Plot the fitted model with the data
plt.figure()
ev.plot_data()
ev.plot_model(color='red')
plt.title('Data and Fitted Model')

plt.show()
