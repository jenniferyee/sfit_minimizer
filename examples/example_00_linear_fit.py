import sfit_minimizer
import matplotlib.pyplot as plt
import numpy as np

"""
Example of fitting a polynomial of the form y = m * x + b to some data.
"""

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


data = np.loadtxt('../data/test_data_10000pts_Poisson.txt', skiprows=2)
initial_guess = [4, 2.1]  # Wrong initial condition
my_func = LinearFunction(data=data)

result = sfit_minimizer.minimize(
    my_func, x0=initial_guess, tol=1e-7,
    options={'step': 'adaptive'}, verbose=True)

print('\nFull Results:')
print(result)
print('\n')

values = result.x
sigmas = result.sigmas
print('fit values: ')
print(values)
print('+/-')
print(sigmas)

my_func.theta = values
my_func.update_all()
print('chi2: ', my_func.chi2)

plt.figure()
plt.title('Values')
plt.errorbar(
    my_func.data[:, 0], my_func.data[:, 1], 
    yerr=my_func.data[:, 2], fmt='o')
x = np.arange(0, 100)
plt.plot(x, values[0] + values[1] * x, color='red', zorder=5)

plt.figure()
plt.title('Residuals')
plt.errorbar(
    my_func.data[:, 0], my_func.residuals,
    yerr=my_func.data[:, 2], fmt='o')
plt.axhline(0, color='red', zorder=5)

plt.show()
