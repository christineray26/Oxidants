import numpy as np
import matplotlib.pyplot as plt

#Calculations based on Hand, 2007 


g_0 = 1.2 #micrometers/yr
t_0 = 1.7 * 10**5 #yrs

#calculate gardening depth 
t = np.arange(0.1, 500, 1) #Myr
t_low = 0.5 * 10**6 #oldest surface age in years
d_g = (10**-6)*g_0*(t*10**6)*(1+((t*10**6)/t_0))**(-0.55) #meters
#print "Gardening depth on Enceladus = %e meters" %dg_enc


#Now calculate gardening volume rate
R = 252 * 10**3
A = 4*np.pi*R**2

r_g = (A*d_g/(t*10**6))* 9.2 * 10**5 #m^3/yr * g/m^3
#print "Gardening rate on Enceladus = %e g/yr" %rg_enc

#And delivery rate of O2
rO2_30 = (r_g / 18.0) * 0.30 #g/yr * mol/g * mol O2/H2O
rO2_20 = (r_g / 18.0) * 0.20 #g/yr * mol/g * mol O2/H2O
rO2_10 = (r_g / 18.0) * 0.10 #g/yr * mol/g * mol O2/H2O
#print "Delivery rate of O2 on Enceladus = %e  mol/yr" %rO2_enc


plt.figure(1, figsize = (12, 6))
plt.plot(t, np.log10(rO2_10), 'k--', label = '10% $O_{2}$')
plt.plot(t, np.log10(rO2_20), 'k-', label = '20% $O_{2}$')
plt.plot(t, np.log10(rO2_30), 'b--', label = '30% $O_{2}$')
plt.xlabel(r'Delivery Period $\tau_{d}$ (Myr)')
plt.ylabel('Log mol $O_{2}$ yr$^{-1}$')
plt.title("Moles $O_{2}$ delivered to the ocean per year by meteoritic gardening")
plt.ylim(7.5, 9.5)
plt.legend(bbox_to_anchor=(1.25,1))
plt.subplots_adjust(left=0.1, right=0.80, top=0.9, bottom=0.1)
#Plot vertical line to indicate upper limit set by surface age
plt.plot([50, 50], [7.6, 9.4], 'k--')
plt.text(55, 9.15, 'surface age = 50 Myr')
#plt.show() 
plt.savefig('O2_Delivery_Gardening.png')








