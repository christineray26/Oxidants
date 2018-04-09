import numpy as np

rep_up = 0.5 #mm/yr
rep_low = 3  * 10**-3 

A = 53456855268.14314 #up to 30 deg from south pole, m^3

rho = 5.2 * 10**5 #porous ice?

M = (1.008*2 + 16)/(6.022*10**23) #g/mol * mol/molecule = g/molecules

r_up = ((rep_up*10**-3)*A*rho)/(M*365*24*3600)
r_low = ((rep_low*10**-3)*A*rho)/(M*365*24*3600)

# (mm/yr) * (10**-3 m/mm) = m/yr
# m/yr * m^2 = m^3/yr
# m^3/yr * g/m^3 = g/yr
# (g/yr)/(g/molecule) = molecules/yr
# molecules/yr * (yr/days) * (day/hrs) * (hr/s) = molecules/s 

