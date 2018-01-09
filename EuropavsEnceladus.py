import numpy as np
import matplotlib.pyplot as plt

#Calculations based on Hand, 2007 


g_0 = 1.2 #micrometers/yr
t_0 = 1.7 * 10**5 #yrs


#Europa

#calculate gardening depth
t_eur = 30 * 10**6 #oldest surface age in years
dg_eur = (10**-6)*g_0*t_eur*(1+(t_eur/t_0))**(-0.55) #meters
print "Gardening depth on Europa = %e meters" %dg_eur

#figure out what their O2 concentration was
R_eur = 1560 * 10**3
A_eur = 4*np.pi*R_eur**2
eta = 5 * 10**9 #mol/yr
eps = (eta * t_eur)/(A_eur * dg_eur)
print 'O2 concentration  = %e mol/m^3'%eps

#Now calculate gardening volume rate
rg_eur = (A_eur*dg_eur/t_eur)* 9.2 * 10**5 #m^3/yr * g/m^3
print "Gardening rate on Europa = %e g/yr" %rg_eur

#And delivery rate of O2
rO2_eur = (rg_eur / 18.0) * 0.0463 #g/yr * mol/g * mol O2/H2O
print "Delivery rate of O2 on Enceladus = %e  mol/yr" %rO2_eur

#Enceladus

#calculate gardening depth 
t_enc = 0.5 * 10**6 #oldest surface age in years
dg_enc = (10**-6)*g_0*t_enc*(1+(t_enc/t_0))**(-0.55) #meters
print "Gardening depth on Enceladus = %e meters" %dg_enc


#Now calculate gardening volume rate
R_enc = 252 * 10**3
A_enc = 4*np.pi*R_enc**2

rg_enc = (A_enc*dg_enc/t_enc)* 9.2 * 10**5 #m^3/yr * g/m^3
print "Gardening rate on Enceladus = %e g/yr" %rg_enc

#And delivery rate of O2
rO2_enc = (rg_enc / 18.0) * 0.20 #g/yr * mol/g * mol O2/H2O
print "Delivery rate of O2 on Enceladus = %e  mol/yr" %rO2_enc
