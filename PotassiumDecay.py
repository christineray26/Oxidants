import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.rcParams['axes.titlesize'] = 24  
plt.rcParams['axes.labelsize'] = 20  
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.titlepad'] = 40 
plt.rcParams['axes.labelpad'] = 16 


#Energies in MeV - from Draganic
E_betaD = 0.455
beta_frac = 0.11
E_gammaD = 1.46
gamma_frac = 0.89
E_avg = (E_betaD*beta_frac*10**6) + (E_gammaD*gamma_frac*10**6) #in eV

ratio = 0.483/0.983

#Energies in MeV from Alexis
E_beta = 1.1760
E_gamma = 0.1566

#Other constants/parameters
lambda_K = 5.4 * 10**-10 #yr^-1
N_A = 6.022*10**23
M_ocean = 10**19 #kg
t = np.arange(0.0, 4.501, 0.001)*10**9
t_plot = np.arange(-4.501, 0.0, 0.001)
t_step = 0.001 * 10**9
N = len(t)

#Calculate K40 concentration 4.5 Gyr ago (K40_0) and as a function of time (K40)
K_conc = 1.0 * 10**-3 #mol/kg H2O
K40_frac = 1.17 * 10**-4
K40_now = K_conc * K40_frac #mol K/kg H2O
#K40_conc = 0.00145 * K_conc ##need to ask chris why N_0's don't match

K40_0 = (K40_now * M_ocean) * np.exp(lambda_K * 4.5*10**9) * N_A #mol/kg H2O * kg H2O * atoms/mol = atoms total
K40 = K40_0 * np.exp(-lambda_K * t)

#Activity
A = lambda_K * K40  # decays/yr

#Calculate absorbed dose rate
D_beta = A * E_beta * 10**6 #decays/yr * MeV/decay *eV/MeV = eV/yr
D_gamma = A * E_gamma * 10**6

#D_tot = D*M_ocean #eV/yr

#Set values for G
#G = np.array([10**-5, 10**-4, 10**-3, 10**-2, 0.1]) #molecules/100eV #check Ben's paper for other values
GH2_beta = 0.6/100.0 #molecules/eV, from Alexis' paper
GH2_gamma = 0.4/100.0


#Calculate rates & integrated production
PH2 = ((D_beta*GH2_beta) + (D_gamma*GH2_gamma))/N_A
PO2 = ratio*PH2
H2 = np.zeros(N)
O2 = np.zeros(N)


for i in range (0, N):
    if i == 0:
        H2[i] = 0.0
        O2[i] = 0.0
    else:
        H2[i] = PH2[i-1] * t_step + H2[i-1]
        O2[i] = PO2[i-1] * t_step + O2[i-1]
        
        
#P_O2 = np.zeros([5, N])
#O2 = np.zeros([5, N])
#t_step = 0.001*10**9

#for i in range(0,5):
#	P_O2[i] = G[i]/100.0 * D_tot
#	for j in range (0,N):
#		if i == 0:
#			O2[i,j] = 0.0
#		else:
#			O2[i,j] = (P_O2[i,j])*t_step + O2[i,j-1]


#Plot O2 production
fig1 = plt.figure(1)
fig1.set_figheight(6.5)
fig1.set_figwidth(10)
plt.clf()
plt.grid()
plt.plot(t_plot, H2, label = "$H_{2}$")
plt.plot(t_plot, O2, label = "$O_{2}$")
plt.xlim(-4.5, 0)
#plt.ylim(0.0, 2.5*10**18)
plt.xlabel('Time from current epoch (Gyr)')
plt.ylabel('Cumulative Production (mol)')
plt.title('Species Produced by $^{40}K$ Decay in the Ocean')
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
plt.legend(bbox_to_anchor=(0.25,1.0), prop={'size': 15})
plt.savefig('PotassiumDecay_AlexisMethod.png')


#Chris's method
#From Draganic et al:
M_D = 1.4 * 10**21 #kg
rH2_D = (3.8 * 10**18)/(2.016 * 100 * 10**6 * M_D)
rO2_D = (3.0 * 10**19)/(32.02 * 100 * 10**6 * M_D)
rH2O2_D = (1.1 * 10**18)/(34.015 * 100 * 10**6 * M_D)

K40_D = (6 * 10**18)/N_A

kH2 = rH2_D/K40_D #yr^-1
kO2 = rO2_D/K40_D
kH2O2 = rH2O2_D/K40_D
kH2C = 2*kO2 + kH2O2


def conc(k , K, t, t_step): 
    rate = k*(K/N_A)
    N = len(t)
    C = np.zeros(N)
    for i in range (0, N):
        if i == 0:
            C[i] = 0.0
        else:
            C[i] = rate[i-1] * t_step + C[i-1]
    return rate, C
    
rH2, H2D = conc(kH2, K40, t, t_step) #mol
rH2C, H2DC = conc(kH2C, K40, t, t_step)
rO2, O2D = conc(kO2, K40, t, t_step)
rH2O2, H2O2 = conc(kH2O2, K40, t, t_step)    

#Write to a file
headers = 'Rate H2        Total H2            Rate O2        Total O2          Rate H2O2      Total H2O2 \n'
f = open('O2Potassium.txt', 'w')
f.write(headers)
for i in range (0,N): 
	f.write(str(rH2[i]) + ', ' + str(H2D[i]) + ', ' + str(rO2[i]) + ', ' + str(O2D[i]) + ', ' + str(rH2O2[i]) + ', ' + str(H2O2[i])+ ' \n')
f.close()

#Plot O2, H2 & H2O2 production
fig2 = plt.figure(2)
fig2.set_figheight(6.5)
fig2.set_figwidth(11)
plt.clf()
plt.grid()
plt.plot(t_plot, O2D, label = "$O_{2}$")
plt.plot(t_plot, H2D, label = "$H_{2}$")
#plt.plot(t_plot, H2DC, label = "$H_{2}$ coupled")
plt.plot(t_plot, H2O2, label = "$H_{2}O_{2}$")
plt.xlim(-4.5, 0)
#plt.ylim(0.0, 2.5*10**18)
plt.xlabel('Time from current epoch (Gyr)')
plt.ylabel('Cumulative Production (mol)')
plt.title('Species Produced by $^{40}K$ Decay in the Ocean')
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
plt.legend(bbox_to_anchor=(0.25,1.0), prop={'size': 18})
plt.savefig('PotassiumDecay_ChrisMethod.png')


#Plot both methods
fig2 = plt.figure(3)
fig2.set_figheight(6.5)
fig2.set_figwidth(11)
plt.clf()
plt.grid()
plt.plot(t_plot, O2D, '-', color = 'royalblue', label = "$O_{2}$, Draganic et al. (1991)")
plt.plot(t_plot, H2D, '-', color = 'darkorange', label = "$H_{2}$, Draganic et al. (1991)")
#plt.plot(t_plot, H2DC, label = "$H_{2}$ coupled")
plt.plot(t_plot, H2O2, color = 'darkturquoise',  label = "$H_{2}O_{2}$, Draganic et al. (1991)")
plt.plot(t_plot, H2, '--', color = 'darkorange', label = "$H_{2}$, Bouquet et al. (2017)")
plt.plot(t_plot, O2, '--', color = 'royalblue', label = "$O_{2}$, Bouquet et al. (2017)")
plt.xlim(-4.5, 0)
plt.ylim(0.0, 1.0*10**17)
plt.xlabel('Time from current epoch (Gyr)')
plt.ylabel('Cumulative Production (mol)')
plt.title('Species Produced by $^{40}$K Decay in the Ocean')
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
plt.legend(bbox_to_anchor=(0.36,1.0), prop={'size': 12.5})
#plt.legend()
plt.savefig('PotassiumDecay_BothMethods.png')

 	
