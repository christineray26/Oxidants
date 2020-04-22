import numpy as np
import matplotlib.pyplot as plt
import os
import sys

##First get parameters of good models

#Read in chi square values
chisqr = np.zeros(4332)
names = list()
i = 0
for fname in os.listdir('../ChiSqrResults/One_Arcsec_VaryNFW/ChiFiles/'):
	#fname = 'c0' + str(i) + '-chi.dat'
	fdir = "../ChiSqrResults/One_Arcsec_VaryNFW/ChiFiles/" + fname
	chi, b, c, d, e, f, g, h, j = np.loadtxt(fdir, unpack = True)
	chisqr[i] = chi
	names.append(fname)
	i = i + 1

#Determine which models are below chi-square cutoff
good_indices = np.where(chisqr < 23.69)
good_chi = chisqr[good_indices]
names = np.array(names)
good_models = names[good_indices]
def remove_end(n):
	return n[:-8]
good_models_new = [remove_end(n) for n in good_models]	

#Write names of good models to file
#f = open('GoodModels.txt', 'w')
#for item in good_models_new:
#	f.write(item)
#	f.write('\n')
#f.close()

#Extract modified NFW paramaters from good models & write to file
#params = open('GoodParams.txt', 'w')
#for name in good_models_new:
#	fname = name + '.dat'
#	with open('../ChiSqrResults/One_Arcsec_VaryNFW/' + str(fname), 'r') as f:
#		for i, x in enumerate(f):
#			if i == 2: 
#				params.write(x)
#			elif i >2:
#				break
#params.close()

#Read in chi-square for second set of models
chisqr_new = np.zeros(4310)
names_new = list()
i = 0
for fname in os.listdir('../ChiSqrResults/One_Arcsec_VaryNFW_New/ChiFiles/'):
	fdir = "../ChiSqrResults/One_Arcsec_VaryNFW_New/ChiFiles/" + fname
	chi, b, c, d, e, f, g, h, j = np.loadtxt(fdir, unpack = True)
	chisqr_new[i] = chi
	names_new.append(fname)
	i = i + 1

#Determine which new models are below chi-square cutoff
good_indices_new = np.where(chisqr_new < 23.69)
good_chi_new = chisqr_new[good_indices_new]
#names_new = np.array(names_new)
#good_models2 = names_new[good_indices_new]
#def remove_end(n):
#	return n[:-8]
#good_models2 = [remove_end(n) for n in good_models2]	

#Write names of good models to file
#f = open('GoodModels_NewImg.txt', 'w')
#for item in good_models2:
#	f.write(item)
#	f.write('\n')
#f.close()

#Pull out NFW parameters into separate arrays
def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue


#Extract modified NFW paramaters from good models & write to file
#params2 = open('GoodParams_NewImg.txt', 'w')
#for name in good_models2:
#	fname = name + '.dat'
#	with open('../ChiSqrResults/One_Arcsec_VaryNFW_New/' + str(fname), 'r') as f:
#		for i, x in enumerate(f):
#			if i == 2: 
#				params2.write(x)
#			elif i >2:
#				break
#params2.close()

##No LOS models
chisqr_noLOS = np.zeros(924)
names_noLOS = list()
i = 0
for fname in os.listdir('../ChiSqrResults/NoLOS/ChiFiles/'):
	fdir = "../ChiSqrResults/NoLOS/ChiFiles/" + fname
	chi, b, c, d, e, f, g, h, j = np.loadtxt(fdir, unpack = True)
	chisqr_noLOS[i] = chi
	names_noLOS.append(fname)
	i = i + 1

#Determine which new models are below chi-square cutoff
good_indices_noLOS = np.where(chisqr_noLOS < 150)
good_chi_noLOS = chisqr_noLOS[good_indices_noLOS]
names_noLOS = np.array(names_noLOS)
good_models_noLOS = names_noLOS[good_indices_noLOS]
def remove_end(n):
	return n[:-8]
good_models_noLOS = [remove_end(n) for n in good_models_noLOS]	

#Write names of good models to file
f = open('GoodModels_NoLOS.txt', 'w')
for item in good_models_noLOS:
	f.write(item)
	f.write('\n')
f.close()

#Extract modified NFW paramaters from good models & write to file
params_noLOS = open('GoodParams_NoLOS.txt', 'w')
for name in good_models_noLOS:
	fname = name + '.dat'
	with open('../ChiSqrResults/NoLOS/DataFiles/' + str(fname), 'r') as f:
		for i, x in enumerate(f):
			if i == 1: 
				params_noLOS.write(x)
			elif i > 1:
				break
params_noLOS.close()

##Now get parameters of original models
params2 = open('AllParams.txt', 'w')
for i in range (0, 9999):
	fname = 'c' + '%05d'%i + '.dat'
	with open('../LensModels/DataFiles/' + str(fname), 'r') as f:
		for i, f in enumerate(f):
			if i == 2:
				params2.write(f)
			elif i >2:
				break
params2.close()

kappa, x ,y, e, te, g, tg, rs, a, b ,z = np.loadtxt(strip_first_col('GoodParams.txt'), unpack = True)

#Parameters of models created from new set of Images
kappa_new, x_new ,y_new, e_new, te_new, g_new, tg_new, rs_new, a_new, b_new ,z_new = np.loadtxt(strip_first_col('GoodParams_NewImg.txt'), unpack = True)

#Parameters of no LOS models
kappa_noLOS, x_noLOS ,y_noLOS, e_noLOS, te_noLOS, g_noLOS, tg_noLOS, rs_noLOS, a_noLOS, b_noLOS = np.loadtxt(strip_first_col('GoodParams_NoLOS.txt'), unpack = True)

kappa2, x2 ,y2, e2, te2, g2, tg2, rs2, a2, b2, z2 = np.loadtxt(strip_first_col('AllParams.txt'), unpack = True)

rs = np.log10(rs)
rs2 = np.log10(rs2)
rs_new = np.log10(rs_new)
rs_noLOS = np.log10(rs_noLOS)

##Triangle plot

kappalim = (0, 1.1)
kappamarkers = (0.0, 0.5, 1.0)

xlim = (-25.0, 15.0)
xmarkers = (-20,-5, 10)

ylim = (-40.0, 40.0)
ymarkers = (-25, 0, 25)

elim = (-0.5, 1.0)
emarkers = (-0.25, 0.25, 0.75)

telim = (-0.1, 0.9)
temarkers = (0.0, 0.4, 0.8)

rslim = (1.0, 3.5)
rsmarkers = (1.5, 2.25, 3)

ylabels = [r"$\kappa_s$", r"$x \, (arsec)$", r"$y \, (arcsec)$", r"$e_c$", r"$e_s$", r"$\log (r_s [kpc])$"]
xlabels = [r"$\log(r_s [kpc])$", r"$e_s$", r"$e_c$", r"$y \, (arcsec)$", r"$x \, (arcsec)$", r"$\kappa_s$"]

#First row
fig = plt.figure(1)
plt.subplot(6,6,1)
l1, = plt.plot(rs2, kappa2, 'k.', markersize = 4.0)
l4, = plt.plot(rs_noLOS, kappa_noLOS, 'g.', markersize = 5.0)
l3, = plt.plot(rs_new, kappa_new, 'b.', markersize = 5.0)
l2, = plt.plot(rs, kappa, 'r.', markersize = 5.0)
plt.xlabel(xlabels[0])
plt.ylabel(ylabels[0])
plt.xlim(rslim)
plt.ylim(kappalim)

plt.subplot(6,6,2)
plt.plot(te2, kappa2, 'k.', markersize = 4.0)
plt.plot(te_noLOS, kappa_noLOS, 'g.', markersize = 5.0)
plt.plot(te_new, kappa_new, 'b.', markersize = 5.0)
plt.plot(te, kappa, 'r.', markersize = 5.0)
plt.xlabel(xlabels[1])
plt.xlim(telim)
plt.ylim(kappalim)

plt.subplot(6,6,3)
plt.plot(e_noLOS, kappa_noLOS, 'g.', markersize = 5.0)
plt.plot(e2, kappa2, 'k.', markersize = 4.0)
plt.plot(e_new, kappa_new, 'b.', markersize = 5.0)
plt.plot(e, kappa, 'r.', markersize = 5.0)
plt.xlabel(xlabels[2])
plt.xlim(elim)
plt.ylim(kappalim)

plt.subplot(6,6,4)
plt.plot(y2, kappa2, 'k.', markersize = 4.0)
plt.plot(y_noLOS, kappa_noLOS, 'g.', markersize = 5.0)
plt.plot(y_new, kappa_new, 'b.', markersize = 5.0)
plt.plot(y, kappa, 'r.', markersize = 5.0)
plt.xlabel(xlabels[3])
#plt.xlim(ylim)
plt.ylim(kappalim)

plt.subplot(6,6,5)
plt.plot(x2, kappa2, 'k.', markersize = 4.0)
plt.plot(x_noLOS, kappa_noLOS, 'g.', markersize = 5.0)
plt.plot(x_new, kappa_new, 'b.', markersize = 5.0)
plt.plot(x, kappa, 'r.', markersize = 5.0)
plt.xlabel(xlabels[4])
#plt.xlim(xlim)
plt.ylim(kappalim)

plt.subplot(6,6,6)
plt.hist(kappa2, bins = 50, range = kappalim, color = 'k')
plt.hist(kappa_noLOS, bins = 30, range = kappalim, color = 'g')
plt.hist(kappa_new, bins = 30, range = kappalim, weights = good_chi_new * 3, color = 'b')
plt.hist(kappa, bins = 30, range = kappalim, weights = good_chi * 3, color = 'r')
plt.xlabel(xlabels[5])

#Second Row
plt.subplot(6,6,7)
plt.plot(rs2, x2, 'k.', markersize = 4.0)
plt.plot(rs_noLOS, x_noLOS, 'g.', markersize = 5.0)
plt.plot(rs_new, x_new, 'b.', markersize = 5.0)
plt.plot(rs, x, 'r.', markersize = 5.0)
plt.xlim(rslim)
plt.ylabel(ylabels[1])

plt.subplot(6,6,8)
plt.plot(te2, x2, 'k.', markersize = 4.0)
plt.plot(te_noLOS, x_noLOS, 'g.', markersize = 5.0)
plt.plot(te_new, x_new, 'b.', markersize = 5.0)
plt.plot(te, x, 'r.', markersize = 5.0)
plt.xlim(telim)

plt.subplot(6,6,9)
plt.plot(e2, x2, 'k.', markersize = 4.0)
plt.plot(e_noLOS, x_noLOS, 'g.', markersize = 5.0)
plt.plot(e_new, x_new, 'b.', markersize = 5.0)
plt.plot(e, x, 'r.', markersize = 5.0)
plt.xlim(elim)

plt.subplot(6,6,10)
plt.plot(y2, x2, 'k.', markersize = 4.0)
plt.plot(y_noLOS, x_noLOS, 'g.', markersize = 5.0)
plt.plot(y_new, x_new, 'b.', markersize = 5.0)
plt.plot(y, x, 'r.', markersize = 5.0)

plt.subplot(6,6,11)
plt.hist(x2, bins = 50, range = xlim, color = 'k')
plt.hist(x_noLOS, bins = 40, range = xlim, color = 'g')
plt.hist(x_new, bins = 40, weights = good_chi_new * 3, range = xlim, color = 'b')
plt.hist(x, bins = 40, weights = good_chi * 3, range = xlim, color = 'r')

#Third Row
plt.subplot(6,6,13)
plt.plot(rs2, y2, 'k.', markersize = 4.0)
plt.plot(rs_noLOS, y_noLOS, 'g.', markersize = 5.0)
plt.plot(rs_new, y_new, 'b.', markersize = 5.0)
plt.plot(rs, y, 'r.', markersize = 5.0)
plt.xlim(rslim)
plt.ylabel(ylabels[2])

plt.subplot(6,6,14)
plt.plot(te2, y2, 'k.', markersize = 4.0)
plt.plot(te_noLOS, y_noLOS, 'g.', markersize = 5.0)
plt.plot(te_new, y_new, 'b.', markersize = 5.0)
plt.plot(te, y, 'r.', markersize = 5.0)
plt.xlim(telim)

plt.subplot(6,6,15)
plt.plot(e2, y2, 'k.', markersize = 4.0)
plt.plot(e_noLOS, y_noLOS, 'g.', markersize = 5.0)
plt.plot(e_new, y_new, 'b.', markersize = 5.0)
plt.plot(e, y, 'r.', markersize = 5.0)
plt.xlim(elim)

plt.subplot(6,6,16)
plt.hist(y2, bins = 50, range = ylim, color = 'k')
plt.hist(y_noLOS, bins = 50, range = ylim, color = 'g')
plt.hist(y_new, bins = 50, weights = good_chi_new * 3, range = ylim, color = 'b')
plt.hist(y, bins = 50, weights = good_chi * 2, range = ylim, color = 'r')
plt.xlim(ylim)

#Fourth Row
plt.subplot(6,6,19)
plt.plot(rs2, e2, 'k.', markersize = 4.0)
plt.plot(rs_noLOS, e_noLOS, 'g.', markersize = 5.0)
plt.plot(rs_new, e_new, 'b.', markersize = 5.0)
plt.plot(rs, e, 'r.', markersize = 5.0)
plt.xlim(rslim)
plt.ylabel(ylabels[3])

plt.subplot(6,6,20)
plt.plot(te2, e2, 'k.', markersize = 4.0)
plt.plot(te_noLOS, e_noLOS, 'g.', markersize = 5.0)
plt.plot(te_new, e_new, 'b.', markersize = 5.0)
plt.plot(te, e, 'r.', markersize = 5.0)
plt.xlim(telim)

plt.subplot(6,6,21)
plt.hist(e2, bins = 50, range = elim, color = 'k')
plt.hist(e_noLOS, bins = 50, range = elim, color = 'g')
plt.hist(e_new, bins = 50, weights = good_chi_new * 3, range = elim, color = 'b')
plt.hist(e, bins = 50, weights = good_chi * 2, range = elim, color = 'r')
plt.xlim(elim)

#Fifth Row
plt.subplot(6,6,25)
plt.plot(rs2, te2, 'k.', markersize = 4.0)
plt.plot(rs_noLOS, te_noLOS, 'g.', markersize = 5.0)
plt.plot(rs_new, te_new, 'b.', markersize = 5.0)
plt.plot(rs, te, 'r.', markersize = 5.0)
plt.xlim(rslim)
plt.ylabel(ylabels[4])

plt.subplot(6,6,26)
plt.hist(te2, bins = 50, range = telim, color = 'k')
plt.hist(te_noLOS, bins = 30, range = telim, color = 'g')
plt.hist(te_new, bins = 30, weights = good_chi_new * 3, range = telim, color = 'b')
plt.hist(te, bins = 30, weights = good_chi * 3, range = telim, color = 'r')
plt.xlim(telim)

#Sixth Row
plt.subplot(6,6,31)
plt.hist(rs2, bins = 50, normed = True, range = rslim, color = 'k')
plt.hist(rs_noLOS, bins = 30, normed = True, range = rslim, color = 'g')
plt.hist(rs_new, bins = 30, normed = True, range = rslim, color = 'b')
plt.hist(rs, bins = 30, normed = True, range = rslim, color = 'r')
plt.ylabel(ylabels[5])


#First row axis labels
ax=plt.subplot(6,6,1)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(rsmarkers)
ax.yaxis.set_ticks(kappamarkers)
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
ax=plt.subplot(6,6,2)
plt.setp(ax.get_yticklabels(), visible=False)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(temarkers)
ax.xaxis.label.set_size(20)
ax=plt.subplot(6,6,3)
plt.setp(ax.get_yticklabels(), visible=False)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(emarkers)
ax.xaxis.label.set_size(20)
ax=plt.subplot(6,6,4)
plt.setp(ax.get_yticklabels(), visible=False)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(ymarkers)
ax.xaxis.label.set_size(20)
ax=plt.subplot(6,6,5)
plt.setp(ax.get_yticklabels(), visible=False)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(xmarkers)
ax.xaxis.label.set_size(20)
ax=plt.subplot(6,6,6)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_ticks(kappamarkers)
ax.xaxis.label.set_size(20)

#Second row axis labels
ax=plt.subplot(6,6,7)
plt.setp(ax.get_xticklabels(), visible=False)
ax.yaxis.set_ticks(xmarkers)
ax.yaxis.label.set_size(20)
ax=plt.subplot(6,6,8)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,9)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,10)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,11)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

#Third row axis labels
ax=plt.subplot(6,6,13)
plt.setp(ax.get_xticklabels(), visible=False)
ax.yaxis.set_ticks(ymarkers)
ax.yaxis.label.set_size(20)
ax=plt.subplot(6,6,14)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,15)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,16)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

#Fourth row axis labels
ax=plt.subplot(6,6,19)
plt.setp(ax.get_xticklabels(), visible=False)
ax.yaxis.set_ticks(emarkers)
ax.yaxis.label.set_size(20)
ax=plt.subplot(6,6,20)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax=plt.subplot(6,6,21)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

#Fifth row axis labels
ax=plt.subplot(6,6,25)
plt.setp(ax.get_xticklabels(), visible=False)
ax.yaxis.set_ticks(temarkers)
ax.yaxis.label.set_size(20)
ax=plt.subplot(6,6,26)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

#Sixth row axis labels
ax=plt.subplot(6,6,31)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.yaxis.set_ticks(rsmarkers)
ax.yaxis.label.set_size(20)


#Adjust spacing in subplots
plt.subplots_adjust(left=0.11, bottom=0.05, right=0.97, top=None, wspace=0.00, hspace=0.00)

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.5   # the amount of height reserved for white space between subplots

fig.legend((l1, l2, l3, l4), ('Original 10,000', 'Image Set 1', 'Image Set 2', 'No LOS'), 'lower right')

plt.show()















