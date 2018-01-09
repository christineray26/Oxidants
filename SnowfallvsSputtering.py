import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint

R = 252 * 10**3 #m

def Afunc_phi(x):
	return np.sin(x)
def Afunc_theta(x):
	return 1
	
Aphi_int = integrate.quad(Afunc_phi, 0, np.pi/6.0)
Atheta_int = integrate.quad(Afunc_theta, 0, 2*np.pi)

A_SF = R**2 * Aphi_int[0] * Atheta_int[0]

SF_up = 0.5 * 10**-3 #m/yr
SF_range = np.asarray([10**-4, 10**-3, 10**-2, 10**-1, 10**0]) * 10**-3

M_H2O = 18.02 #g/mole
N_A = 6.022 * 10**23

r_SF = ((A_SF * SF_up * 5.2 * 10**5)/M_H2O) * N_A/(365 * 24 * 3600) #m^2 * m/yr * g/m^3 = g/yr * mol/g * molecules/mol = molecules/yr * yr/days * day/hrs * hr/seconds = molecules/s
r_SFrange = ((A_SF * SF_range * 5.2 * 10**5)/M_H2O) * N_A/(365 * 24 * 3600) 





