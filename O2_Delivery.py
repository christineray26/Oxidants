import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['axes.labelsize'] = 17  
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlepad'] = 40 
plt.rcParams['axes.labelpad'] = 15 

### Gardening rate - consider everywhere except south pole ###

g_0 = 1.2 #micrometers/yr
t_0 = 1.7 * 10**5 #yrs

#calculate gardening depth 

###Use this for varying delivery period###
#t_d = np.arange(0.1, 600, 1) #Myr

##Calculate constraint from mass of ice in south pole divided by plume eruption rate##
#integrate to get vol of south pole - assume 60 deg south latitude
def func_R(x):
	return x**2
def func_phi(x):
	return np.sin(x)
def func_theta(x):
	return 1
R_int = integrate.quad(func_R, 237, 252)
phi_int = integrate.quad(func_phi, 0 , np.pi/3.0)
theta_int = integrate.quad(func_theta, 0, 2*np.pi)

V = R_int[0]*phi_int[0]*theta_int[0] #Volume of south polar ice in km^3

#Convert volume to mass
rho = 917 * 1000**3 #kg/km^3
M_SP = rho*V #kg

#divide by eruption rate and convert to years
r_erupt = 200.0 #kg/s
t_up = (M_SP/r_erupt)/(3600*24*365.0)


###Use this for fixed delivery period###
#t_d = 0.50  #Myr
t_d = t_up

d_g = (10**-6)*g_0*(t_d*10**6)*(1+((t_d*10**6)/t_0))**(-0.55) #meters
#print "Gardening depth on Enceladus = %e meters" %dg_enc

##Now calculate gardening volume rate##

#take south pole region to span 60 deg south latitude
#integrate to get surface area of south pole
R = 252 * 10**3 #m

def Afunc_phi(x):
	return np.sin(x)
def Afunc_theta(x):
	return 1
Aphi_int = integrate.quad(Afunc_phi, 0, np.pi/3.0)
Atheta_int = integrate.quad(Afunc_theta, 0, 2*np.pi)

A_SF = R**2 * Aphi_int[0] * Atheta_int[0]

#total surface area of enceladus
A = 4*np.pi*R**2

#total surface area of "old" surface
A_g =  A - A_SF

#gardening rate
r_g = (A_g*d_g/(t_d*10**6))* 9.2 * 10**5 #m^3/yr * g/m^3
#print "Gardening rate on Enceladus = %e g/yr" %r_g

#And delivery rate of O2
rO2_30 = (r_g / 18.0) * 0.30 #g/yr * mol/g * mol O2/H2O
rO2_20 = (r_g / 18.0) * 0.20 #g/yr * mol/g * mol O2/H2O
rO2_10 = (r_g / 18.0) * 0.10 #g/yr * mol/g * mol O2/H2O
rO2_01 = (r_g / 18.0) * 0.01 #g/yr * mol/g * mol O2/H2O
#print "Delivery rate of O2 from gardening = %e  mol/yr" %rO2_20



### Snowfall rate --consider steady-state system at south pole ###
r_SF = A_SF * (0.5 * 10**-3) * 5.2 * 10**5 #m^3/yr * g/m^3 = g/yr, 0.5 comes from article Hunter sent

#print "Snowfall rate on Enceladus = %e g/yr" %r_SF
###check on limit from H2 paper- source # 76 ###
#contact CDA instrument people on snowfall rate #

#18.01 g/mol, 0.2 mol O2/mol H2O
rO2_SF20 = (r_SF/18.0) * 0.2 #g/yr * mol_H2O /g * mol_O2/mol_H2O = mol_O2/yr
rO2_SF30 = (r_SF/18.0) * 0.3 #g/yr * mol_H2O /g * mol_O2/mol_H2O = mol_O2/yr
rO2_SF10 = (r_SF/18.0) * 0.1 #g/yr * mol_H2O /g * mol_O2/mol_H2O = mol_O2/yr
rO2_SF1 = (r_SF/18.0) * 0.01 #g/yr * mol_H2O /g * mol_O2/mol_H2O = mol_O2/yr
#print "Delivery rate of O2 from snowfall = %e  mol/yr" %rO2_SF


### Add for total delivery rate & integrate over enceladus lifetime ###
rO2_20 = rO2_20 + rO2_SF20
rO2_10 = rO2_10 + rO2_SF10
rO2_30 = rO2_30 + rO2_SF30
rO2_01 = rO2_01 + rO2_SF1
#print "Total delivery rate of O2 = %e mol/yr" %rO2


##For variable delivery period##
#O2_20 = rO2_20 * 4.5 * 10**9
#O2_10 = rO2_10 * 4.5 * 10**9
#O2_30 = rO2_30 * 4.5 * 10**9
#O2_01 = rO2_01 * 4.5 * 10**9
#print "Total amount of oxygen delivered to ocean over lifetime = %e moles" %O2

### Plot total amount of available O2 as a function of delivery period ###
#plt.figure(1)
#f, axs = plt.subplots(2,2,figsize=(15,8))
#plt.clf()
#plt.cla()

#01% O2
#sub1 = plt.subplot(2,2,1)
#plt.plot(t_d, O2_01, 'k-')
#plt.title ('1% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(1.296*10**20, 1.302*10**20)
#plt.setp(sub1.get_xticklabels(), visible=False)
#plt.ylabel('Mol $O_{2}$')
##Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [1.2965*10**20, 1.301*10**20], 'k--')
#plt.text(55, 1.300*10**20, 'surface age = 50 Myr')
##Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**20, 1.300*10**20], 'k--')
#plt.text(415, 1.2993*10**20, 'constraint from')
#plt.text(415, 1.299*10**20, 'plume eruption rate') 

#10% O2
#sub2 = plt.subplot(2,2,2)
#plt.plot(t_d, O2_10, 'k-')
#plt.title ('10% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(1.296*10**21, 1.302*10**21)
#plt.setp(sub2.get_xticklabels(), visible=False)
##Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [1.2965*10**21, 1.301*10**21], 'k--')
#plt.text(55, 1.300*10**21, 'surface age = 50 Myr')
##Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**21, 1.300*10**21], 'k--')
#plt.text(415, 1.2993*10**21, 'constraint from')
#plt.text(415, 1.299*10**21, 'plume eruption rate') 

#20% O2
#sub3 = plt.subplot(2,2,3)
#plt.plot(t_d, O2_20, 'k-')
#plt.title ('20% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(2.593*10**21, 2.603*10**21)
#plt.xlabel(r'Delivery Period $\tau_{d}$ (Myr)')
#plt.ylabel('Mol $O_{2}$')
#Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [2.5925*10**21, 2.600*10**21], 'k--')
#plt.text(55, 1.300*10**21, 'surface age = 50 Myr')
#Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**21, 1.300*10**21], 'k--')
#plt.text(415, 1.2993*10**21, 'constraint from')
#plt.text(415, 1.299*10**21, 'plume eruption rate') 

#30% O1
#sub4 = plt.subplot(2,2,4)
#plt.plot(t_d, O2_30, 'k-')
#plt.title ('30% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(3.890*10**21, 3.901*10**21)
#plt.xlabel(r'Delivery Period $\tau_{d}$ (Myr)')

#plt.suptitle("Total amount of $O_{2}$ available from surface radiolysis after 4.5 Gyr")

#plt.legend(bbox_to_anchor=(1.25,1))
#plt.subplots_adjust(left=0.1, right=0.80, top=0.9, bottom=0.1)

#plt.savefig('TotalO2_SurfaceRad_tau.png')


### Plot total O2 production by surface radiolysis with fixed tau_d ###
t_plot = np.arange(-4.5, 0.1, 0.01)
t = np.arange(0, 4.6, 0.01) * 10**9
O2_20 = rO2_20 * t
O2_10 = rO2_10 * t
O2_01 = rO2_01 * t
O2_30 = rO2_30 * t

fig2 = plt.figure(2)
fig2.set_figheight(6.5)
fig2.set_figwidth(11)
plt.clf()
plt.plot(t_plot, O2_01, 'b--', label = '01% $O_{2}$')
plt.plot(t_plot, O2_10, 'k--', label = '10% $O_{2}$')
plt.plot(t_plot, O2_20, 'k-', label = '20% $O_{2}$')
#plt.plot(t_plot, O2_30, 'b--', label = '30% $O_{2}$')
plt.xlabel('Time from Current Epoch (Gyr)')
plt.ylabel('Mol $O_{2}$')
plt.xlim(-4.5, 0)
plt.title(r"Total amount of $O_{2}$ available from surface radiolysis") #$\tau_d$ = %d Myr" %(t_up*10**-6)
plt.legend(bbox_to_anchor=(0.2,1), prop={'size': 15})
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
#plt.show()
plt.savefig('TotalO2_SurfaceRad_time.png')


### Now add in subsurface radiolysis, assume 2H20 = 2H2 + O2 ###

#After 4.5 Gyr
H2 = 1.9 * 10**18
O2_sub = 0.5*H2

#O2_tot01 = O2_01 + O2_sub
#O2_tot10 = O2_10 + O2_sub
#O2_tot20 = O2_20 + O2_sub
#O2_tot30 = O2_30 + O2_sub


###Plot total O2 production as a function of delivery period after 4.5 Gyr###
#plt.figure(3, figsize = (13, 6))
#plt.clf()
#plt.cla()
#f, axs = plt.subplots(2,2,figsize=(15,8))

#01% O2
#sub1 = plt.subplot(2,2,1)
#plt.plot(t_d, O2_tot01, 'k-')
#plt.title ('1% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(1.305*10**20, 1.315*10**20)
#plt.setp(sub1.get_xticklabels(), visible=False)
#plt.ylabel('Mol $O_{2}$')
#Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [1.2965*10**20, 1.301*10**20], 'k--')
#plt.text(55, 1.300*10**20, 'surface age = 50 Myr')
#Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**20, 1.300*10**20], 'k--')
#plt.text(415, 1.2993*10**20, 'constraint from')
#plt.text(415, 1.299*10**20, 'plume eruption rate') 

#10% O2
#sub2 = plt.subplot(2,2,2)
#plt.plot(t_d, O2_tot10, 'k-',)
#plt.title ('10% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(1.296*10**21, 1.302*10**21)
#plt.setp(sub2.get_xticklabels(), visible=False)
#Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [1.2965*10**21, 1.301*10**21], 'k--')
#plt.text(55, 1.300*10**21, 'surface age = 50 Myr')
#Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**21, 1.300*10**21], 'k--')
#plt.text(415, 1.2993*10**21, 'constraint from')
#plt.text(415, 1.299*10**21, 'plume eruption rate') 

#20% O2
#sub3 = plt.subplot(2,2,3)
#plt.plot(t_d, O2_tot20, 'k-')
#plt.title ('20% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(2.593*10**21, 2.603*10**21)
#plt.xlabel(r'Delivery Period $\tau_{d}$ (Myr)')
#plt.ylabel('Mol $O_{2}$')
#Plot vertical line to indicate upper limit set by surface age
#plt.plot([50, 50], [2.5925*10**21, 2.600*10**21], 'k--')
#plt.text(55, 1.300*10**21, 'surface age = 50 Myr')
#Plot vertical line to indicate limit set by plume eruption rate 
#plt.plot([(t_up*10**-6), (t_up*10**-6)], [1.2965*10**21, 1.300*10**21], 'k--')
#plt.text(415, 1.2993*10**21, 'constraint from')
#plt.text(415, 1.299*10**21, 'plume eruption rate') 

#30% O1
#sub4 = plt.subplot(2,2,4)
#plt.plot(t_d, O2_tot30, 'k-')
#plt.title ('30% $O_{2}$') 
#plt.xlim(0, 600)
#plt.ylim(3.890*10**21, 3.901*10**21)
#plt.xlabel(r'Delivery Period $\tau_{d}$ (Myr)')

#plt.suptitle("Total amount of $O_{2}$ available from surface & subsurface radiolysis after 4.5 Gyr")

#plt.legend(bbox_to_anchor=(1.25,1))
#plt.subplots_adjust(left=0.1, right=0.80, top=0.9, bottom=0.1)
#Plot vertical line to indicate upper limit set by surface age
#plt.savefig('TotalO2_tau.png')



###Oxygen Sinks###

#Start by calculating upper limits


##Once one reaction consumes all of the reactant, remaining O2 gets divided up into the next two, and then into just CO2
#C + O2 = CO2
#Greenalite: 4Fe3Si2O5(OH)4 + 10H2O + 3O2 = 12Fe(OH)3 + 8SiO2 
#Pyrrhotite: Fe(0.875)S  2.313 H2O + 2.156 O2 = 0.875 Fe(OH)3 + SO4 + 2H+
#Magnetite: 4Fe3O4  + 18H2O + O2 = 12Fe(OH)3

M_rock = 6 * 10**22 #g

##Organic Carbon##
M_C = 0.2*M_rock #try it for 2% & 20% , section 4.2.1 of H2 paper
mol_C = 12.011
C = M_C/mol_C
CO2_up = C

##Reduced Hydrous Case##
M_Gr = 0.2975 * M_rock #mass
mol_Gr = 3*55.845 + 2*28.086 + 9*16.0 + 4*1.008
Gr = M_Gr/mol_Gr
FeOH3Gr_up = 3*Gr 

M_PyR = 0.1927 * M_rock
mol_PyR = 0.875*55.845 + 32.065
PyR = M_PyR/mol_PyR
SO4R_up = 1*PyR
FeOH3PyR_up = 0.875*PyR

##Oxidized Hydrous Case##
M_Mag = 0.1911 * M_rock
mol_Mag = 3*55.845 + 16*4
Mag = M_Mag/mol_Mag
FeOH3Mag_up = 3*Mag

M_PyO = 0.1987 * M_rock
mol_PyO = 0.875*55.845 + 32.065
PyO = M_PyO/mol_PyO
SO4O_up = 1*PyO
FeOH3PyO_up = 0.875*PyO

#Production plots over time
N = len(t)
t_step = 0.01*10**9
#Amount of oxygen produced during each timestep
O2_step = rO2_20 * t_step
 
###Reduced Hydrous Case###

##Greenalite oxidation: 4Fe3Si2O5(OH)4 + 10H2O + 3O2 = 12Fe(OH)3 + 8SiO2 ##

#amount of O2 required to react with all greenalite:
O2req_Gr = (3.0/4.0)*Gr # if 1/3 of the O2 produced is less than this, O2 is the limiting reagent

#Fe(OH)3 from greenalite oxidation
FeOH3Gr = np.zeros(N)
O2tot_Gr = np.zeros(N)

for i in range (0,N):
	if i == 0:
		O2tot_Gr[i] = 0.0
	else:
		O2tot_Gr[i] = (O2_step/3.0) + O2tot_Gr[i-1]
	if O2tot_Gr[i] < O2req_Gr: #then oxygen is the limiting reagent
		FeOH3Gr_val = 4.0 * O2tot_Gr[i]
	else: #then greenalite is the limiting reagent
		FeOH3Gr_val = FeOH3Gr_up
	FeOH3Gr[i] = FeOH3Gr_val

#Figure out when this reaction ends (Greenalite is used up first)
endrxn1 = ((np.where(O2tot_Gr > O2req_Gr))[0])[0] #check on this


##Pyrrhotite oxidation: Fe(0.875)S  2.313 H2O + 2.156 O2 = 0.875 Fe(OH)3 + SO4 + 2H+##
#amount of O2 required to react with all pyrrhotite
O2req_PyR = 2.156*PyR # if 1/3 of the O2 produced is less than this, O2 is the limiting reagent

#Fe(OH)3 & SO4 from pyrrhotite oxidation
FeOH3PyR = np.zeros(N)
SO4R = np.zeros(N)

#Step 1 - Greenalite oxidation is still occurring 
O2tot_PyR = np.zeros(N)

for i in range (0,(endrxn1-1)):
	if i == 0:
		O2tot_PyR[i] = 0.0
	else:
		O2tot_PyR[i] = (O2_step/3.0) + O2tot_PyR[i-1]
	if O2tot_PyR[i] < O2req_PyR: #then oxygen is the limiting reagent
		FeOH3PyR_val = (0.875/2.156)*O2tot_PyR[i]
		SO4_val = (1.0/2.156)*O2tot_PyR[i]
	else: #then pyrrhotite is the limiting reagent
		FeOH3PyR_val = FeOH3PyR_up
		SO4_val = SO4R_up
	FeOH3PyR[i] = FeOH3PyR_val
	SO4R[i] = SO4_val

#Step 2 - greenalite oxidation ended
for i in range ((endrxn1-1),N):
	O2tot_PyR[i] = (O2_step/2.0) + O2tot_PyR[i-1]
	if O2tot_PyR[i] < O2req_PyR: #then oxygen is the limiting reagent
		FeOH3PyR_val = (0.875/2.156)*O2tot_PyR[i]
		SO4_val = (1.0/2.156)*O2tot_PyR[i]
	else: #then pyrrhotite is the limiting reagent
		FeOH3PyR_val = FeOH3PyR_up
		SO4_val = SO4R_up
	FeOH3PyR[i] = FeOH3PyR_val
	SO4R[i] = SO4_val


#Figure out when second reaction ends (Pyrrhotite oxidation ends)
endrxn2 = ((np.where(O2tot_PyR > O2req_PyR))[0])[0] #check on this

	
#CO2
N = len(t)
CO2 = np.zeros(N)
O2req_C = C
O2tot_C = np.zeros(N)



#Step 1 - Greenalite oxidation is still occurring 

#Calculate available O2
for i in range (0,(endrxn1-1)):
	if i == 0:
		O2tot_C[i] = 0.0
	else:
		O2tot_C[i] = (O2_step/3.0) + O2tot_C[i-1]
	if O2tot_C[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_C[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2[i] = CO2_val

#Step 2 - greenalite oxidation ended
for i in range ((endrxn1-1),(endrxn2-1)):
	O2tot_C[i] = (O2_step/2.0) + O2tot_C[i-1]
	if O2tot_C[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_C[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2[i] = CO2_val

#Step 3 - pyrrhotite oxidation ended	
for i in range ((endrxn2-1), N):
	O2tot_C[i] = O2_step + O2tot_C[i-1]
	if O2tot_C[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_C[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2[i] = CO2_val
	
#Add FeOH3 from both reactions for total
FeOH3R = FeOH3Gr + FeOH3PyR

fig4 = plt.figure(4)
fig4.set_figheight(6.5)
fig4.set_figwidth(11)
plt.clf()
plt.plot(t_plot, CO2, 'k--', label = '$CO_{2}$ (Oxidation of Organic Carbon)')
#plt.text(-1.5, 0.9*10**21, r'$C + O_{2} \rightarrow CO_{2}$', fontsize = 12)
plt.plot(t_plot, FeOH3R, 'g--', label = '$Fe(OH)_{3}$ (Greenalite and Pyrrhotite Oxidation)')
#plt.text(-3.0, 0.35*10**21, r'$4Fe_{3}Si_{2}O_{5}(OH)_{4} + 10H_{2}O + 3O_{2} \rightarrow 12Fe(OH)_{3} + 8SiO_{2}$',  fontsize = 12, color = 'g')
plt.plot(t_plot, SO4R, 'b--', label = '$SO_{4}$ (Pyrrhotite Oxidation)')
#plt.text(-3.1, 0.075*10**21, r'$Fe_{0.875}S + 2.313H_{2}O + 2.156O_{2} \rightarrow 0.875Fe(OH)_{3} + SO_{4} + 2H^{+}$', fontsize = 12, color = 'b')
#Plot vertical lines to mark end of reactions - arrows?
#plt.plot([t_plot[endrxn1-1], t_plot[endrxn1-1]], [-0.05*10**21, 0.8*10**21], 'k:')
#plt.plot([t_plot[endrxn2-1], t_plot[endrxn2-1]], [0.05*10**21, 0.9*10**21], 'k:')
plt.xlabel('Time from Current Epoch (Gyr)')
plt.ylabel('Total Moles')
plt.xlim(-4.5, 0)
plt.title(r"Compounds Produced from O$_{2}$ Oxidation, Reduced Hydrous Case")
plt.legend(bbox_to_anchor=(1.0,0.7), prop={'size': 14})
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
#plt.savefig('Geochem_Reduced.png')

##Oxidized Hydrous Case##

##Magnetite Oxidation: 4Fe3O4  + 18H2O + O2 = 12Fe(OH)3##

#Amount of O2 required to react with all magnetite
O2req_mag = (1.0/4.0)*Mag
O2tot_mag = np.zeros(N)

#FeOH3 from magnetite oxidation
FeOH3mag = np.zeros(N)

for i in range (0,N):
	if i == 0:
		O2tot_mag[i] = 0.0
	else:
		O2tot_mag[i] = (O2_step/3.0) + O2tot_mag[i-1]
	if O2tot_mag[i] < O2req_mag: #then oxygen is the limiting reagent
		FeOH3mag_val = 12.0 * O2tot_mag[i]
	else: #then magnetite is the limiting reagent
		FeOH3mag_val = FeOH3Mag_up
	FeOH3mag[i] = FeOH3mag_val

endrxn1_O = ((np.where(O2tot_mag > O2req_mag))[0])[0] #check on this

##Pyrrhotite oxidation: Fe(0.875)S  2.313 H2O + 2.156 O2 = 0.875 Fe(OH)3 + SO4 + 2H+##

#amount of O2 required to react with all pyrrhotite
O2req_PyO = 2.156*PyO # if 1/3 of the O2 produced is less than this, O2 is the limiting reagent
O2tot_PyO = np.zeros(N)

#Fe(OH)3 & SO4 from pyrrhotite oxidation
FeOH3PyO = np.zeros(N)
SO4O = np.zeros(N)

#Step 1 - Magnetite oxidation is still occurring 
O2tot_PyR = np.zeros(N)

for i in range (0,(endrxn1_O-1)):
	if i == 0:
		O2tot_PyO[i] = 0.0
	else:
		O2tot_PyO[i] = (O2_step/3.0) + O2tot_PyO[i-1]
	if O2tot_PyO[i] < O2req_PyO: #then oxygen is the limiting reagent
		FeOH3PyO_val = (0.875/2.156)*O2tot_PyO[i]
		SO4_val = (1.0/2.156)*O2tot_PyO[i]
	else: #then pyrrhotite is the limiting reagent
		FeOH3PyO_val = FeOH3PyO_up
		SO4_val = SO4O_up
	FeOH3PyO[i] = FeOH3PyO_val
	SO4O[i] = SO4_val

#Step 2 - magnetite oxidation ended
for i in range ((endrxn1_O-1),N):
	O2tot_PyO[i] = (O2_step/2.0) + O2tot_PyO[i-1]
	if O2tot_PyO[i] < O2req_PyO: #then oxygen is the limiting reagent
		FeOH3PyO_val = (0.875/2.156)*O2tot_PyO[i]
		SO4_val = (1.0/2.156)*O2tot_PyO[i]
	else: #then pyrrhotite is the limiting reagent
		FeOH3PyO_val = FeOH3PyO_up
		SO4_val = SO4O_up
	FeOH3PyO[i] = FeOH3PyO_val
	SO4O[i] = SO4_val


#Figure out when second reaction ends (Pyrrhotite oxidation ends)
endrxn2_O = ((np.where(O2tot_PyO > O2req_PyO))[0])[0] #check on this


#CO2
CO2_O = np.zeros(N)
O2req_CO = C
O2tot_CO = np.zeros(N)


#Step 1 - Magnetite oxidation is still occurring 

#Calculate available O2
for i in range (0,(endrxn1_O-1)):
	if i == 0:
		O2tot_CO[i] = 0.0
	else:
		O2tot_CO[i] = (O2_step/3.0) + O2tot_CO[i-1]
	if O2tot_CO[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_CO[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2_O[i] = CO2_val

#Step 2 - magnetite oxidation ended
for i in range ((endrxn1_O-1),(endrxn2_O-1)):
	O2tot_CO[i] = (O2_step/2.0) + O2tot_CO[i-1]
	if O2tot_CO[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_CO[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2_O[i] = CO2_val

#Step 3 - pyrrhotite oxidation ended	
for i in range ((endrxn2_O-1), N):
	O2tot_CO[i] = O2_step + O2tot_CO[i-1]
	if O2tot_CO[i] < C: #then oxygen is the limiting reagent
		CO2_val = O2tot_CO[i]
	else: #then carbon is the limiting reagent
		CO2_val = CO2_up
	CO2_O[i] = CO2_val

#Add FeOH3 from both reactions for total
FeOH3O = FeOH3mag + FeOH3PyO

#Plot reactions for oxidized case
fig5 = plt.figure(5)
fig5.set_figheight(6.5)
fig5.set_figwidth(11)
plt.clf()
plt.plot(t_plot, CO2_O, 'k--', label = '$CO_{2}$ (Oxidation of Organic Carbon)')
#plt.text(-1.5, 0.9*10**21, r'$C + O_{2} \rightarrow CO_{2}$', fontsize = 12)
plt.plot(t_plot, FeOH3O, 'g--', label = '$Fe(OH)_{3}$ (Magnetite and Pyrrhotite Oxidation)')
#plt.text(-3.0, 0.35*10**21, r'$4Fe_{3}Si_{2}O_{5}(OH)_{4} + 10H_{2}O + 3O_{2} \rightarrow 12Fe(OH)_{3} + 8SiO_{2}$',  fontsize = 12, color = 'g')
plt.plot(t_plot, SO4O, 'b--', label = '$SO_{4}$ (Pyrrhotite Oxidation)')
plt.xlabel('Time from Current Epoch (Gyr)')
plt.ylabel('Total Moles')
plt.xlim(-4.5, 0)
plt.title(r"Compounds Produced from O$_{2}$ Oxidation, Oxidized Hydrous Case")
plt.legend(bbox_to_anchor=(0.0,1.0), prop={'size': 13})
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
plt.savefig('Geochem_Oxidized.png')



###Now Plot Concentrations###

#Reduced Hydrous Case#
fig4 = plt.figure(4)
fig4.set_figheight(6.5)
fig4.set_figwidth(11)
plt.clf()
plt.plot(t_plot, CO2, 'k--', label = '$CO_{2}$ (Oxidation of Organic Carbon)')
#plt.text(-1.5, 0.9*10**21, r'$C + O_{2} \rightarrow CO_{2}$', fontsize = 12)
plt.plot(t_plot, FeOH3R, 'g--', label = '$Fe(OH)_{3}$ (Greenalite and Pyrrhotite Oxidation)')
#plt.text(-3.0, 0.35*10**21, r'$4Fe_{3}Si_{2}O_{5}(OH)_{4} + 10H_{2}O + 3O_{2} \rightarrow 12Fe(OH)_{3} + 8SiO_{2}$',  fontsize = 12, color = 'g')
plt.plot(t_plot, SO4R, 'b--', label = '$SO_{4}$ (Pyrrhotite Oxidation)')
#plt.text(-3.1, 0.075*10**21, r'$Fe_{0.875}S + 2.313H_{2}O + 2.156O_{2} \rightarrow 0.875Fe(OH)_{3} + SO_{4} + 2H^{+}$', fontsize = 12, color = 'b')
#Plot vertical lines to mark end of reactions - arrows?
#plt.plot([t_plot[endrxn1-1], t_plot[endrxn1-1]], [-0.05*10**21, 0.8*10**21], 'k:')
#plt.plot([t_plot[endrxn2-1], t_plot[endrxn2-1]], [0.05*10**21, 0.9*10**21], 'k:')
plt.xlabel('Time from Current Epoch (Gyr)')
plt.ylabel('Total Moles')
plt.xlim(-4.5, 0)
plt.title(r"Compounds Produced from O$_{2}$ Oxidation, Reduced Hydrous Case")
plt.legend(bbox_to_anchor=(1.0,0.7), prop={'size': 14})
plt.subplots_adjust(left=0.12, right=0.93, top=0.8, bottom=0.15)
#plt.savefig('Geochem_Reduced.png')

