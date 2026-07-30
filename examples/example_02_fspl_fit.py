"""
Example of fitting a point lens to some data using MulensModel. Includes finite
source effects.
"""
import os.path

import matplotlib.pyplot as plt
import MulensModel as mm

import sfit_minimizer

if int(mm.__version__.split('.')[0]) < 3:
    print('Finite Source gradient will be implemented in MulensModel v3.')

mm.utils.MAG_ZEROPOINT = 18.

# Read in the data:
data_path = os.path.join(sfit_minimizer.DATA_PATH, 'MMTest')
datafiles = ['FSPL_Obs_1_I.pho', 'FSPL_Obs_2_V.pho']
datasets = []
for i, filename in enumerate(datafiles):
    if i == 0:
        band = 'I'
    else:
        band = 'V'

    data = mm.MulensData(
        file_name=os.path.join(data_path, filename), phot_fmt='mag',
        bandpass=band)

    datasets.append(data)

# Create the model and event objects
gammas = {'I': 0.44,
          'V': 0.72,
          'H': 0.26}
model = mm.Model({'t_0': 2451697.2, 'u_0': 0.00600, 't_E': 25.00000,
                  'rho': 0.0065})
n_t_star = 100.
t_star = model.parameters.rho * model.parameters.t_E
model.set_magnification_methods([
    model.parameters.t_0 - n_t_star * t_star,
    'finite_source_LD_Yoo04',
    model.parameters.t_0 + n_t_star * t_star])
for band, value in gammas.items():
    model.set_limb_coeff_gamma(band, value)

event = mm.Event(datasets=datasets, model=model)

# Setup the fitting
parameters_to_fit = ['t_0', 'u_0', 't_E', 'rho']
initial_guess = [2451697.19995,  0.00600, 25.00000,  0.00650, 1.30000,  0.00000,
                 1.00000,  0.00000]

my_func = sfit_minimizer.mm_funcs.PointLensSFitFunction(event, parameters_to_fit)

# Do the fit
result = sfit_minimizer.minimize(
    my_func, x0=initial_guess, tol=1e-4,
    options={'step': 'adaptive'}, verbose=True)

# Print the results
print('Full Results:')
print(result)

values = result.x
sigmas = result.sigmas
print('results: ')
print(values)
print('+/-')
print(sigmas)

my_func.update_all(values)
print('chi2: ', my_func.chi2)

# Plot results
my_func.event.plot(subtract_2450000=True)
plt.show()
