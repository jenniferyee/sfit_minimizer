"""
Example of fitting a point lens to some data using MulensModel.
"""
import os.path

import matplotlib.pyplot as plt
import MulensModel as mm

import sfit_minimizer

# Read in the data:
data_path = os.path.join(sfit_minimizer.DATA_PATH, 'MMTest')
datafiles = ['PSPL_1_Obs_1.pho', 'PSPL_1_Obs_2.pho']
datasets = []
for filename in datafiles:
    data = mm.MulensData(
        file_name=os.path.join(data_path, filename), phot_fmt='mag')
    datasets.append(data)

datasets.append(datasets[-1])

# Create the model and event objects
model = mm.Model({'t_0': 8650., 'u_0': 0.30000, 't_E': 25.00000})
event = mm.Event(datasets=datasets, model=model)

# Setup the fitting
parameters_to_fit = ['t_0', 'u_0', 't_E']
initial_guess = []
for key in parameters_to_fit:
    if key == 't_E':
        initial_guess.append(model.parameters.parameters[key].value)
    else:
        initial_guess.append(model.parameters.parameters[key])

for i in range(len(datasets)):
    initial_guess.append(1.0)
    initial_guess.append(0.0)

my_func = sfit_minimizer.mm_funcs.PointLensSFitFunction(event, parameters_to_fit)

# Do the fit
result = sfit_minimizer.minimize(
    my_func, x0=initial_guess, tol=1e-5,
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
