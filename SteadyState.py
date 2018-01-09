import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['axes.labelsize'] = 17  
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlepad'] = 40 
plt.rcParams['axes.labelpad'] = 15 

A = 1.7290320655709074e-12
B = 26.503957180196277
C = 8.9265336203209493e-13
l = 5.400000000000001e-10


#Before 10^7 yr delivery period
t = np.arange(0, 20, 0.001)*10**6
t_plot = np.arange(0, 20, 0.001)
SS_before = (C*np.exp(-l*t))/B

fig1 = plt.figure(1)
fig1.set_figheight(6.5)
fig1.set_figwidth(10)
plt.clf()
plt.plot(t_plot, SS_before, 'b-')
plt.xlabel('Total Elapsed Time (Myr)')
plt.ylabel('$O_{2}$ Concentration (mol/kg $H_{2}O$)')
plt.xlim(0.0, 20)
plt.ylim(3.330*10**-14, 3.37*10**-14)
plt.title(r"Steady State $O_{2}$ Concentration in the Ocean") #$\tau_d$ = %d Myr" %(t_up*10**-6)
plt.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.15)
plt.grid()
plt.savefig('SteadyState_InitialDelivery.png')


#After 10^7 yr delivery period
t = np.arange(0, 4.501, 0.001)*10**9
t_plot = np.arange(-4.5, 0.001, 0.001)
SS_after = (A+C*np.exp(-l*t))/B

fig2 = plt.figure(2)
fig2.set_figheight(6.5)
fig2.set_figwidth(10)
plt.clf()
plt.plot(t_plot, SS_after, 'b-')
plt.xlabel('Time from Current Epoch (Gyr)')
plt.ylabel('$O_{2}$ Concentration (mol/kg $H_{2}O$)')
plt.xlim(-4.5, 0.0)
plt.ylim(0.65*10**-13, 1.0*10**-13)
plt.title(r"Steady State $O_{2}$ Concentration in the Ocean") #$\tau_d$ = %d Myr" %(t_up*10**-6)
plt.subplots_adjust(left=0.12, right=0.9, top=0.8, bottom=0.15)
plt.grid()
plt.savefig('SteadyState.png')
