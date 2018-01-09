import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.rcParams['axes.titlesize'] = 25  
plt.rcParams['axes.labelsize'] = 23  
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlepad'] = 20 
plt.rcParams['axes.labelpad'] = 6 


pH = np.array([8, 9, 10, 11, 12])
pOH = 14 - pH #should actually be ~14.7 for 0 degrees
OH = 10.0**(-pOH)

###Reaction 1: 4Fe(II) + O2 + 4H+ = 4Fe(III) + 2H2O###
#t_O2 = 4/(k * Fe * OH^2)

k1 = 10**(14.3) #table 7 in Millero, 1987 
#Ask Chris what iron concentration range is appropriate
#10**-9 to 10**-3, log scale
Fe1 = np.arange(-9.0, -2.9, 0.1) #molar 

#Calculate t_O2 for each pH (seconds?)
N = len(Fe1)
t1_O2 = np.zeros((5, N))
#t_O2 in minutes/ 60 / 24 to days
t1_O2[0] = (4.0/(k1 * 10**Fe1 * OH[0]**2))#/(60*24*365)
t1_O2[1] = (4.0/(k1 * 10**Fe1 * OH[1]**2))#/(60*24*365)
t1_O2[2] = (4.0/(k1 * 10**Fe1 * OH[2]**2))#/(60*24*365)
t1_O2[3] = (4.0/(k1 * 10**Fe1 * OH[3]**2))#/(60*24*365)
#t_O2[4] = (4.0/(k1 * 10**Fe1 * OH[4]**2))#/(60*24*365)

#Pl2ot O2 consumption time as a function of iron concentration
fig1 = plt.figure(1)
fig1.set_figheight(7)
fig1.set_figwidth(12)
plt.subplots_adjust(left=0.1, bottom=None, right=0.82, top=None, wspace=None, hspace=None)
plt.clf()
#plt.grid()
plt.plot(Fe1, np.log10(t1_O2[0]), label = 'pH = 8.0')
plt.plot(Fe1, np.log10(t1_O2[1]), label = 'pH = 9.0')
plt.plot(Fe1, np.log10(t1_O2[2]), label = 'pH = 10.0')
plt.plot(Fe1, np.log10(t1_O2[3]), label = 'pH = 11.0')
#plt.plot(Fe1, np.log10(t1_O2[4]), label = 'pH = 12.0')
#Plot horizontal lines for time labels
plt.plot([-9, -3], [np.log10(60), np.log10(60)], 'k--')
plt.text(-3.8, 2.1,'1 Hour', fontsize = 15 )
plt.plot([-9, -3], [np.log10(60*24*365), np.log10(60*24*365)], 'k--')
plt.text(-3.8, 6.1,'1 Year', fontsize = 15 )
plt.xlabel('log[[Fe$^{+2}$](M)]')
plt.ylabel(r'log[$\tau_{O2}$ (minutes)]')
plt.xlim(-9.0, -3.0)
plt.title(r'Time Required to Reduce [$O_{2}$] by $\frac{1}{e}$')
plt.legend(bbox_to_anchor=(0.3,0.35), prop={'size': 17})
plt.savefig('tauO2_iron.png')

#Using O2 production over time, plot 

#tau = residence time, O2 divided by rate, time to decrease concentration by a factor of 1/e

###Reaction 2: H2S + 2O2 = SO4 + 2H+ ###
#t_O2 = 1/2k[H2S]

k2 = 10**(1.435) #minutes I hope???
#Ask Chris what iron concentration range is appropriate
#10**-9 to 10**-3, log scale
H2S = np.arange(-9.0, -2.9, 0.1) #molar 

N = len(H2S)
t2_O2 = 1.0/(2.0 * k2 * 10**H2S)

fig2 = plt.figure(2)
fig2.set_figheight(7)
fig2.set_figwidth(12)
plt.subplots_adjust(left=0.1, bottom=None, right=0.82, top=None, wspace=None, hspace=None)
plt.clf()
#plt.grid()
plt.plot(H2S, np.log10(t2_O2))
#plt.plot(H2S, np.log10(t2_O2[4]), label = 'pH = 12.0')
#Plot horizontal lines for time labels
plt.plot([-9, -3], [np.log10(60), np.log10(60)], 'k--')
plt.text(-3.8, 2.1,'1 Hour', fontsize = 15 )
plt.plot([-9, -3], [np.log10(60*24*365), np.log10(60*24*365)], 'k--')
plt.text(-3.8, 6.1,'1 Year', fontsize = 15 )
plt.xlabel('log[[$H_{2}S$](M)]')
plt.ylabel(r'log[$\tau_{O2}$ (minutes)]')
plt.xlim(-9.0, -3.0)
plt.title(r'Time Required to Reduce [$O_{2}$] by $\frac{1}{e}$')
#plt.legend(bbox_to_anchor=(1.005,1.0), prop={'size': 14})
plt.savefig('tauO2_sulfide.png')



