import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

T = 273 #K maybe change to -13 later?
P = 1 #bar, check this is okay for LV

#Concentrations (mol/L)
pH = 6.2
H = 10**(-pH)
SO4 = .0584 # +/- 2.3
CO2 = .008860 # +/- 190
H2 = 0.00001047 #=/- 0.02
CH4 = 0.000001 # <0.000001
N2O = 0.0000588 
NH4 = .0038852 # 43.0
NO2 = 0.0000237 # 1.0
NO3 = 0.0009044 # 30.0
DOC = 0.0482 # +/- 9700
#HCO3 = 
Glucose = 0.7*DOC
H2O = 0.87

FeS2 = 1.0
FeS = 1.0
Fe3O4 = 1.0
Fe2O3 = 1.0
Fe2 = 1.0

N2 = 0.0009


#FeS2 (assume activity is 1) + NO3(or other oxidant) = SO4 + Fe + NH4
#or FeS (assume 1)

#also iron oxidation 
#Fe2+ + oxidant = Fe3O4 (or Fe2O3)  + reductant

#look for any papers that talk about perchlorate (ClO4)

###Reactions with all compounds in paper###
#1 eps proteobacteria - hydrogen oxidation
r1 = '*4H2 + NO3 + 2H+ = NH4 + 3H2O          '
K1 = 131.768
Q1 = np.log((NH4*(H2O**3))/((H2**4)*NO3*(H**2))) 

#2
r2 = 'CH4 + NO3 + 2H+ = CO2 + NH4 + H20      '
K2 = 103.695
Q2 = np.log((CO2*NH4*H2O)/(CH4*NO3*(H**2)))

#3 methanogenesis
r3 = '4H2 + CO2 = CH4 + 2H2O                 '
K3 = 26.965
Q3 = np.log((CH4*(H2O**2))/((H2**4)*CO2))

#4
r4 = 'NH4 + CO2 + H2O = NO3 + CH4 + 2H+      '
K4 = -104.803
Q4 = np.log((NO3*CH4*(H**2))/(NH4*CO2*H2O))

#5 Denitrification 1
r5 = '*2NO3 + 4H+ = 2NO2 + 2H2O              '
K5 = 60.192
Q5 = np.log(((H2O**2)*(NO2**2))/((H**4)*(NO3**2)))


###Reactions with N2###
#6 eps-proteobacteria - hydrogen oxidation
r6 = '*5H2 + 2NO3 + 2H+ = N2 + 6H2O          '
K6 = 231.420
Q6 = np.log((N2*(H2O**6))/((H2**5)*(NO3**2)*(H**2)))

#7 
r7 = '5CH4 + 8NO3 + 8H+ = 5CO2 + 4N2 + 14H2O '
K7 = 790.854
Q7 = np.log(((CO2**5)*(N2**4)*(H2O**14))/((H**8)*(NO3**8)*(CH4**5)))

#8 
r8 = '5NH4 + 3NO3 = 4N2 + 2H+ + 9H2O         '
K8 = 266.838
Q8 = np.log(((H2O**9)*(H**2)*(N2**4))/((NO3**3)*(NH4**5)))

#9
r9 = '3H2 + N2 + 2H+ = 2NH4                  '
K9 = 32.117
Q9 = np.log((NH4**2)/((H**2)*N2*(H2**3)))

#10
r10 = '3CH4 + 4N2 + 8H+ + 6H2O = 8NH4 + 3CO2  '
K10 = 44.246
Q10 = np.log(((NH4**8)*(CO2**3))/((H2O**6)*(H**8)*(N2**4)*(CH4**3)))

#11
r11 = '9H2O + 4N2 + 2H+ = 5NH4 + 3NO3         '
K11 = -266.084
Q11 = np.log(((NO3**3)*(NH4**5))/((H**2)*(N2**4)*(H2O**9)))

#12
r12 = '8NH4 + 3CO2 = 4N2 + 3CH4 + 8H+ + 6H2O  '
K12 = -47.572
Q12 = np.log(((H2O**6)*(H**8)*(N2**4)*(CH4**3))/((NH4**8)*(CO2**3)))

#13
r13 = '4N2 + 5CO2 + 14H2O = 8NO3 + 5CH4 + 8H+ '
K13 = -790.854
Q13 = np.log(((H**8)*(NO3**8)*(CH4**5))/((CO2**5)*(N2**4)*(H2O**14)))


#7 Dentrification 2
#r7 = '2NO2 + 4H+ = 2NO + 2H2O'
#K7 = 43.866
#Q7 = np.log(((H2O**2)*(NO**2))/((H**4)*(NO2**2)))

#8 Denitrification 3
#r8 = '2NO + 2H+ = N2O + H2O'
#K8 = 59.180
#Q8 = np.log((H2O*N2O)/((H**2)*(NO**2)))


##Iron Oxidation Reactions

#14
r14 = '3Fe2+ + 2CO2 + 8H+ = Fe3O4 + 2CH4      '
K14 = 5.147
Q14 = np.log(((CH4**2)*Fe3O4)/((H**8)*(CO2**2)*(Fe2**3)))

#15
r15 = '4Fe2+ + 3CO2 + 12H+ = 2Fe2O3 + 3CH4    '
K15 = 18.889
Q15 = np.log(((CH4**3)*(Fe2O3**2))/((H**12)*(CO2**3)*(Fe2**4)))

#16
r16 = '9Fe2+ + 4NO3 + 16H+ = 3Fe3O4 + 4NH4    '
K16 = 395.038
Q16 = np.log(((NH4**4)*(Fe3O4**3))/((H**16)*(NO3**4)*(Fe2**9)))

#17
r17 = '2Fe2+ + NO3 + 4H+ = Fe2O3 + NH4        '
K17 = 100.339
Q17 = np.log((Fe2O3*NH4)/((H**4)*NO3*(Fe2**2)))

##Sulfur Oxidation Reactions

#18
r18 = '*3FeS2 + 8NO3 + 32H+ = 6SO4 + 3Fe2+ + 8NH4'
K18 = 769.500
Q18 = np.log(((SO4**6)*(Fe2**3)*(NH4**8))/((H**32)*(NO3**8)*(FeS2**3)))

#19
r19 = '*FeS2 + 4CO2 + 16H+ = 2SO4 + Fe2+ + 4CH4'
K19 = 14.114
Q19 = np.log(((CH4**4)*Fe2*(SO4**2))/((H**16)*(CO2**4)*FeS2))

rxns = np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19])
K = np.array([K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16, K17, K18, K19])
Q = np.array([Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19])
   
def A(T, K, Q):
	R = 8.31446 #J mol^-1 K^-1
	A = (2.3026 * R * T * (K - Q))/1000.0
	return A
aff = A(T, K, Q)

#Print to file
fmt = lambda x : "%03d" % x + str(x%1)[1:5]

f = open('ListofReactions.txt', 'w')
print >> f, 'Reaction' + '                                   ', 'log(K)' + '      ', 'log(Q)' + '     ', 'Affinity (kJ/mol)'
for i in range (0,19): 
	print >> f, rxns[i] + '     ', fmt(K[i]) + '     ', fmt(Q[i]) + '    ', fmt(aff[i])
f.close()



