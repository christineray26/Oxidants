import numpy as np
import matplotlib.pyplot as plt
import os

#Create list of finished file names
names = list()

#Pull out chi-square values for models & write file names to list
chisqr = np.zeros(4332)
i = 0
for fname in sorted(os.listdir('ChiSqrResults/One_Arcsec_VaryNFW/ChiFiles/')):
	fdir = "./ChiSqrResults/One_Arcsec_VaryNFW/ChiFiles/" + fname
	chi, b, c, d, e, f, g, h, j = np.loadtxt(fdir, unpack = True)
	chisqr[i] = chi
	i = i + 1
	names.append(fname.replace('-chi.dat', ''))


##Create arrays of kappa and gamma values for each image
#source 1, image 1
kap11 = np.zeros(4332)
gam11 = np.zeros(4332)
#source 1, image 2
kap12 = np.zeros(4332)
gam12 = np.zeros(4332)
#source 1, image 3
kap13 = np.zeros(4332)
gam13 = np.zeros(4332)
#source 2, image 1
kap21 = np.zeros(4332)
gam21 = np.zeros(4332)
#source 2, image 2
kap22 = np.zeros(4332)
gam22 = np.zeros(4332)
#source 2, image 3
kap23 = np.zeros(4332)
gam23 = np.zeros(4332)
#source 3, image 1
kap31 = np.zeros(4332)
gam31 = np.zeros(4332)
#source 3, image 2
kap32 = np.zeros(4332)
gam32 = np.zeros(4332)
#source 3, image 3
kap33 = np.zeros(4332)
gam33 = np.zeros(4332)

n = 0
names2 = list()


for model in names:
	fname = open("./Magnifications/All/" + str(model) + '-kap.dat')
	for i, lines in enumerate(fname):
		if i == 2:
			src1img1 = lines.split()
			kap11[n] = src1img1[3]
			gam11[n] = src1img1[4]
		if i == 3:
			src1img2 = lines.split()
			kap12[n] = src1img2[3]
			gam12[n] = src1img2[4]
		if i == 4:
			src1img3 = lines.split()
			kap13[n] = src1img3[3]
			gam13[n] = src1img3[4]
		if i == 6:
			src2img1 = lines.split()
			kap21[n] = src2img1[3]
			gam21[n] = src2img1[4]
		if i == 7:
			src2img2 = lines.split()
			kap22[n] = src2img2[3]
			gam22[n] = src2img2[4]
		if i == 8:
			src2img3 = lines.split()
			kap23[n] = src2img3[3]
			gam23[n] = src2img3[4]
		if i == 10:
			src3img1 = lines.split()
			kap31[n] = src3img1[3]
			gam31[n] = src3img1[4]
		if i == 11:
			src3img2 = lines.split()
			kap32[n] = src3img2[3]
			gam32[n] = src3img2[4]
		if i == 12:
			src3img3 = lines.split()
			kap33[n] = src3img3[3]
			gam33[n] = src3img3[4]
	n = n + 1
	names2.append([fname])
	fname.close()

#Magnifications, source 1
mag11 = 1.0/((1 - kap11)**2-gam11**2)
mag12 = 1.0/((1 - kap12)**2-gam12**2)
mag13 = 1.0/((1 - kap13)**2-gam13**2)

#Magnifications, source 2
mag21 = 1.0/((1 - kap21)**2-gam21**2)
mag22 = 1.0/((1 - kap22)**2-gam22**2)
mag23 = 1.0/((1 - kap23)**2-gam23**2)

#Magnifications, source 3
mag31 = 1.0/((1 - kap31)**2-gam31**2)
mag32 = 1.0/((1 - kap32)**2-gam32**2)
mag33 = 1.0/((1 - kap33)**2-gam33**2)


## Magnification for "true" model
fname = open("./Magnifications/GoodModels/c02973-kap.dat")
for i, lines in enumerate(fname):
	if i == 2:
		src1img1 = lines.split()
		kap11_true = float(src1img1[3])
		gam11_true = float(src1img1[4])
	if i == 3:
		src1img2 = lines.split()
		kap12_true = float(src1img2[3])
		gam12_true = float(src1img2[4])
	if i == 4:
		src1img3 = lines.split()
		kap13_true = float(src1img3[3])
		gam13_true = float(src1img3[4])
	if i == 6:
		src2img1 = lines.split()
		kap21_true = float(src2img1[3])
		gam21_true = float(src2img1[4])
	if i == 7:
		src2img2 = lines.split()
		kap22_true = float(src2img2[3])
		gam22_true = float(src2img2[4])
	if i == 8:
		src2img3 = lines.split()
		kap23_true = float(src2img3[3])
		gam23_true = float(src2img3[4])
	if i == 10:
		src3img1 = lines.split()
		kap31_true = float(src3img1[3])
		gam31_true = float(src3img1[4])
	if i == 11:
		src3img2 = lines.split()
		kap32_true = float(src3img2[3])
		gam32_true = float(src3img2[4])
	if i == 12:
		src3img3 = lines.split()
		kap33_true= float(src3img3[3])
		gam33_true = float(src3img3[4])
fname.close()

#Magnifications, source 1
mag11_true = 1.0/((1 - kap11_true)**2-gam11_true**2)
mag12_true = 1.0/((1 - kap12_true)**2-gam12_true**2)
mag13_true = 1.0/((1 - kap13_true)**2-gam13_true**2)

#Magnifications, source 2
mag21_true = 1.0/((1 - kap21_true)**2-gam21_true**2)
mag22_true = 1.0/((1 - kap22_true)**2-gam22_true**2)
mag23_true = 1.0/((1 - kap23_true)**2-gam23_true**2)

#Magnifications, source 3
mag31_true = 1.0/((1 - kap31_true)**2-gam31_true**2)
mag32_true = 1.0/((1 - kap32_true)**2-gam32_true**2)
mag33_true = 1.0/((1 - kap33_true)**2-gam33_true**2)

##Take ratios between calculated magnifications and true magnifications
#First source
mag11_rat = np.log10(mag11/mag11_true)
mag12_rat = np.log10(mag12/mag12_true)
mag13_rat = np.log10(mag13/mag13_true)
#Second source
mag21_rat = mag21/mag21_true
mag22_rat = mag22/mag22_true
mag23_rat = mag23/mag23_true
#Third source
mag31_rat = mag31/mag31_true
mag32_rat = mag32/mag32_true
mag33_rat = mag33/mag33_true

#Concatenate all good models into 420 x 9 array and sort by chi-square value
i = np.where(chisqr < 100)
models = np.zeros([420,10])
models[:,0] = chisqr[i]
models[:,1] = mag11_rat[i]
models[:,2] = mag12_rat[i]
models[:,3] = mag13_rat[i]
models[:,4] = mag21_rat[i]
models[:,5] = mag22_rat[i]
models[:,6] = mag23_rat[i]
models[:,7] = mag31_rat[i]
models[:,8] = mag32_rat[i]
models[:,9] = mag33_rat[i]
models_sorted = np.asarray(sorted(models, key = lambda x: x[0]))


#Bin up models and calculate mean, standard dev for each bin
bins_chi = np.zeros((8,52))
centers = np.zeros(8)
bin_sizes = np.zeros(8)

#Src 1
bins_mag11 = np.zeros((8,52))
bins_mag12 = np.zeros((8,52))
bins_mag13 = np.zeros((8,52))
mu_11 = np.zeros(8)
sig_11 = np.zeros(8)
mu_12 = np.zeros(8)
sig_12 = np.zeros(8)
mu_13 = np.zeros(8)
sig_13 = np.zeros(8)

#Src 2
bins_mag21 = np.zeros((8,52))
bins_mag22 = np.zeros((8,52))
bins_mag23 = np.zeros((8,52))
mu_21 = np.zeros(8)
sig_21 = np.zeros(8)
mu_22 = np.zeros(8)
sig_22 = np.zeros(8)
mu_23 = np.zeros(8)
sig_23 = np.zeros(8)

#Src 3
bins_mag31 = np.zeros((8,52))
bins_mag32 = np.zeros((8,52))
bins_mag33 = np.zeros((8,52))
mu_31 = np.zeros(8)
sig_31= np.zeros(8)
mu_32 = np.zeros(8)
sig_32 = np.zeros(8)
mu_33 = np.zeros(8)
sig_33 = np.zeros(8)

lower = 0
upper = 52
for i in range (0, 8):
	bins_chi[i] = models_sorted[lower:upper,0]
	centers[i] = np.median(bins_chi[i])
	bin_sizes[i] = models_sorted[upper-1, 0] - centers[i]
	#Source 1
	bins_mag11[i] = models_sorted[lower:upper,1]
	bins_mag12[i] = models_sorted[lower:upper,2]
	bins_mag13[i] = models_sorted[lower:upper,3]
	mu_11[i] = np.mean(bins_mag11[i])
	sig_11[i] = np.std(bins_mag11[i])
	mu_12[i] = np.mean(bins_mag12[i])
	sig_12[i] = np.std(bins_mag12[i])
	mu_13[i] = np.mean(bins_mag13[i])
	sig_13[i] = np.std(bins_mag13[i])
	#Source 2
	bins_mag21[i] = models_sorted[lower:upper,4]
	bins_mag22[i] = models_sorted[lower:upper,5]
	bins_mag23[i] = models_sorted[lower:upper,6]
	mu_21[i] = np.mean(bins_mag21[i])
	sig_21[i] = np.std(bins_mag21[i])
	mu_22[i] = np.mean(bins_mag22[i])
	sig_22[i] = np.std(bins_mag22[i])
	mu_23[i] = np.mean(bins_mag23[i])
	sig_23[i] = np.std(bins_mag23[i])
	#Source 3
	bins_mag31[i] = models_sorted[lower:upper,7]
	bins_mag32[i] = models_sorted[lower:upper,8]
	bins_mag33[i] = models_sorted[lower:upper,9]
	mu_31[i] = np.mean(bins_mag31[i])
	sig_31[i] = np.std(bins_mag31[i])
	mu_32[i] = np.mean(bins_mag32[i])
	sig_32[i] = np.std(bins_mag32[i])
	mu_33[i] = np.mean(bins_mag33[i])
	sig_33[i] = np.std(bins_mag33[i])
	lower = lower + 52
	upper = upper + 52


#Plot distribution of ratios for good models & points for mu, sigma
#Source 1
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(chisqr, mag11_rat, 'b.', label = 'Image 1')
plt.errorbar(centers, mu_11, xerr = bin_sizes, yerr = sig_11, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (-1, 1)
plt.ylabel('Model Value/True Value')
plt.title('Magnification Ratios for Source #1')
plt.legend()
plt.subplot(3,1,2)
plt.plot(chisqr, mag12_rat, 'c.', label = 'Image 2')
plt.errorbar(centers, mu_12, xerr = bin_sizes, yerr = sig_12, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (-1, 1)
plt.ylabel('Model Value/True Value')
plt.legend()
plt.subplot(3,1,3)
plt.plot(chisqr, mag13_rat, 'g.', label = 'Image 3')
plt.errorbar(centers, mu_13, xerr = bin_sizes, yerr = sig_13, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (-1, 1)
plt.xlabel('$\chi^{2}$')
plt.ylabel('Model Value/True Value')
plt.legend()

#Source 2
plt.figure(2)
plt.subplot(3,1,1)
plt.semilogy(chisqr, mag21_rat, 'b.', label = 'Image 1')
plt.errorbar(centers, mu_21, xerr = bin_sizes, yerr = sig_21, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0, 10)
plt.ylabel('Model Value/True Value')
plt.title('Magnification Ratios for Source #2')
plt.legend()
plt.subplot(3,1,2)
plt.semilogy(chisqr, mag22_rat, 'c.', label = 'Image 2')
plt.errorbar(centers, mu_22, xerr = bin_sizes, yerr = sig_22, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0.1, 10)
plt.ylabel('Model Value/True Value')
plt.legend()
plt.subplot(3,1,3)
plt.semilogy(chisqr, mag23_rat, 'g.', label = 'Image 3')
plt.errorbar(centers, mu_23, xerr = bin_sizes, yerr = sig_23, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0.1, 10)
plt.xlabel('$\chi^{2}$')
plt.ylabel('Model Value/True Value')
plt.legend()

#Source 3
plt.figure(3)
plt.subplot(3,1,1)
plt.semilogy(chisqr, mag31_rat, 'b.', label = 'Image 1')
plt.errorbar(centers, mu_31, xerr = bin_sizes, yerr = sig_31, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0, 10)
plt.ylabel('Model Value/True Value')
plt.title('Magnification Ratios for Source #3')
plt.legend()
plt.subplot(3,1,2)
plt.semilogy(chisqr, mag32_rat, 'c.', label = 'Image 2')
plt.errorbar(centers, mu_32, xerr = bin_sizes, yerr = sig_32, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0.1, 10)
plt.ylabel('Model Value/True Value')
plt.legend()
plt.subplot(3,1,3)
plt.semilogy(chisqr, mag33_rat, 'g.', label = 'Image 3')
plt.errorbar(centers, mu_33, xerr = bin_sizes, yerr = sig_33, fmt = 'sk', ecolor = 'k', fillstyle = 'none', markersize = 9, mew = 1.3)
plt.xlim(0, 100)
plt.ylim (0.1, 10)
plt.xlabel('$\chi^{2}$')
plt.ylabel('Model Value/True Value')
plt.legend()
plt.show()


	#img, x, y, kappa, gamma, theta = np.loadtxt(fname, unpack = True)
	#chisqr[i] = chi


