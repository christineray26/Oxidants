import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


T = 273
P = 1 #bar

#Concentrations (mol/L)
pH = np.array([9.0, 11.0])
H = 10**(-pH)
H2O = 1.0


#Reductants
H2 = np.array([1 * 10**-4,  2 * 10**-7]) #at pH 9 and pH 11
H2S = np.array([1.4 * 10**-7,  2 * 10**-10])
HS = np.array([10**-5.4,  10**-5.4])
CH4 = np.array([3 * 10**-5,  4 * 10**-8])
Fe2 = np.array([10**-8,  10**-8])
FeS2 = np.array([1.0, 1.0])


#Oxidants
O2 = np.array([9.3 ** -15, 2.4**-17])
SO4 = np.array([1.7 * 10**-3, 4.5 * 10**-6]) 
CO2 = np.array([7 * 10**-5,  1 * 10**-7])
#FeOOH = np.array([5.5 * 10**-4,  1.4 * 10**-2])
FeOOH = np.array([1.0, 1.0])



#Reactions
#K values calculated for T = 0.01C and P = 1.0bar

#1 methanogenesis
r1 = '4H2 + CO2 = CH4 + 2H2O                 '
K1 = 37.44
#Q1 = np.log10((CH4*(H2O**2))/((H2**4)*CO2))
Q1 = np.log10(CH4) + 2*np.log10(H2O) - np.log10(CO2) - 4*np.log10(H2)

#2 sulfide oxidation
r2 = 'H2S + 2O2 = SO4 + 2H2O                 '
K2 = 234.754
#Q2 = np.log10((SO4*(H2O**2))/((O2**2)*H2S))
Q2 = 2*np.log10(H2O) + np.log10(SO4) - 2*np.log10(O2) - np.log10(H2S)

#3 sulfide oxidation
r3 = 'HS- + 2O2 = SO4 + H+'
K3 = 
Q3 = np.log10(SO4) + np.log10(H) - 2*np.log10(O2) - np.log10(HS)

#4 iron sulfide oxidation
r4 = 'FeS2 + 3.5O2 + H2O = 2SO4 + Fe(II) + 2H+'
K4 = 
Q4 = 2*np.log10(SO4) + np.log10(Fe2) + 2*np.log10(H) - np.log10(H2O) - 3.5*np.log10(O2) - np.log10(FeS2)

#5 hydrogen oxidation
r5 = '2H2 + O2 = 2H2O                        '
K5 = 100.957
#Q5 = np.log10((H2O**2)/((H2**2)*O2))
Q5 = 2*np.log10(H2O) - 2*np.log10(H2) - np.log10(O2)

#6 sulfate reduction
r6 = '4H2 + SO4 + 2H+ = H2S + 4H2O           '
K6 = 46.347
#Q6 = np.log10((H2S*(H2O**4))/((H2**4)*SO4*(H**2)))
Q6 = np.log10(H2S) + 4*np.log10(H2O) - np.log10(SO4) - 2*np.log10(H) - 4*np.log10(H2)

#7 sulfate reduction
r7 = '4H2 + SO4 + H+ = HS- + 4H2O'
K7 = 
Q7 = np.log10(HS) + 4*np.log10(H2O)  - np.log10(H) - np.log10(SO4) - 4*np.log10(H2)

#8 anaerobic oxidation of methane
r8 = 'CH4 + SO4 + 2H+ = CO2 + H2S + 2H2O     '
K8 = 19.79
#Q8 = np.log10((CO2*H2S*(H2O**2))/(CH4*SO4*(H**2)))
Q8 = np.log10(CO2) + np.log10(H2S) + 2*np.log10(H2O) - np.log10(CH4) - np.log10(SO4) - 2*np.log10(H)

#9 aerobic oxidation of methane
r9 = 'CH4 + 2O2 = CO2 + 2H2O                 '
K9 = 164.473
#Q9 = np.log10((CO2*(H2O**2))/((O2**2)*CH4))
Q9 = np.log10(CO2) + 2*np.log10(H2O) - np.log10(CH4) - 2*np.log10(O2)

#10 reduction of ferric iron


rxns = np.array([r1, r2, r3, r4, r5, r6])
K = np.array([K1, K2, K3, K4, K5, K6])

#pH9
Q_9 = np.array([Q1[0], Q2[0], Q3[0], Q4[0], Q5[0], Q6[0]])

#pH11
Q_11 = np.array([Q1[1], Q2[1], Q3[1], Q4[1], Q5[1], Q6[1]])
   
def A(T, K, Q):
	R = 8.31446 #J mol^-1 K^-1
	A = (2.3026 * R * T * (K - Q))/1000.0
	return A
aff9 = A(T, K, Q_9)
aff11 = A(T, K, Q_11)


#Now calculate power
CO2_mix = np.array([0.003, 0.008])
CH4_mix = np.array([0.001, 0.003])
H2_mix = np.array([0.004, 0.014])
H2S_mix = np.array([10.0/10**6, 10.0/10**6])
#H2S_mix = CH4_mix
plume_v = 200.0 #kg/s

#convert mixing ratios to annual production rates
def mix_to_rate(mix, v):
    return v*(1000.0/18.015)*(3600*24*365)*mix
    
r_CO2 = mix_to_rate(CO2_mix, plume_v)
r_CH4 = mix_to_rate(CH4_mix, plume_v)
r_H2 = mix_to_rate(H2_mix, plume_v)
r_O2 = 5.4 * 10**6
r_H2S = mix_to_rate(H2S_mix, plume_v)

#calculate power
def power(aff, rate):
    return (aff/1000.0)*(rate/(3600*24*365))
    
#methanogenesis: 4H2 + CO2 = CH4 + 2H2O, LR = H2
P1_9low = power(aff9[0], r_H2[0])
P1_9up = power(aff9[0], r_H2[1])
P1_11low = power(aff11[0], r_H2[0])
P1_11up = power(aff11[0], r_H2[1])

#AOM: CH4 + SO4 + 2H+ = CO2 + H2S + 2H2O, LR = CH4
P2_9low = power(aff9[1], r_CH4[0])
P2_9up = power(aff9[1], r_CH4[1])
P2_11low = power(aff11[1], r_CH4[0])
P2_11up = power(aff11[1], r_CH4[1])

#Sulfate Reduction: 4H2 + SO4 + 2H+ = H2S + 4H2O , LR = H2
P3_9low = power(aff9[2], r_H2[0])
P3_9up = power(aff9[2], r_H2[1])
P3_11low = power(aff11[2], r_H2[0])
P3_11up = power(aff11[2], r_H2[1])

#4 hydrogen oxidation, 2H2 + O2 = 2H2O  
P4_9 = power(aff9[3], r_O2)
P4_11 = power(aff11[3], r_O2)

#5 methane oxidation, CH4 + 2O2 = CO2 + 2H2O
P5_9 = power(aff9[4], r_O2)
P5_11 = power(aff11[4], r_O2)

#6 sulfide oxidation, H2S + 2O2 = SO4 + 2H2O
P6_9 = power(aff9[5], r_O2)
P6_11 = power(aff11[5], r_O2)

P_9low = np.array([P1_9low, P2_9low, P3_9low, P4_9, P5_9, P6_9])
P_9up = np.array([P1_9up, P2_9up, P3_9up, P4_9, P5_9, P6_9])
P_11low = np.array([P1_11low, P2_11low, P3_11low, P4_11, P5_11, P6_11])
P_11up = np.array([P1_11up, P2_11up, P3_11up, P4_11, P5_11, P6_11])

#Print to file
fmt = lambda x : "%08.3f" % x #+ str(x%1)[0:5]

headers = 'Reaction                                    log(K)      log(Q)      Affinity   Power_Lower  Power_Upper \n'
units = '                                                                    (kJ/mol)      (J/s)        (J/s)    \n'

f = open('EnceladusOxidationReactions.txt', 'w')
f.write('Concentrations:\n')
f.write('      pH = 9          pH = 11\n')
f.write('H2    1E-4            2E-7   \n')
f.write('O2    1E-14           1E-14  \n')
f.write('SO4   2E-3            2E-3   \n')
f.write('H2S   1.4E-7          2E-10  \n')
f.write('CO2   7E-5            1E-7   \n')
f.write('CH4   3E-5            4E-8   \n')
f.write('  \n')
f.write(headers)
f.write(units)
f.write(' \n')
f.write('pH = 9')
f.write(' \n')
for i in range (0,6): 
	f.write(rxns[i] + '     ' + fmt(K[i]) + '    ' + fmt(Q_9[i]) + '    ' + fmt(aff9[i]) + '    ' + fmt(P_9low[i]) + '     ' +fmt(P_9up[i]) + ' \n')
f.write(' \n')
f.write('pH = 11')
f.write(' \n')
for i in range (0,6): 
    f.write(rxns[i] + '     ' + fmt(K[i]) + '    ' + fmt(Q_11[i]) + '    ' + fmt(aff11[i]) + '    ' + fmt(P_11low[i]) + '     ' +fmt(P_11up[i]) + ' \n')	
f.close()


