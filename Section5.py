import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np


x0 = 325  # in mm

x0err = 0.001
xerr = 0.001
derr = np.sqrt(x0err*x0err + xerr*xerr)
cor = 0.

n = []
inct = []
x = []


def ntrials(n):
	return len(n)

def builderrs(num,ns,ds,Cs):
	w = 4. * derr * derr
	derrs = []
	Cerrs = []
	Cfracs = []
	i = 0
	while i < num:
		derrs.append(derr)
		ntarg = ns[i]
		dtarg = ds[i]
		Cerrs.append(Cs[i]*np.sqrt((1./ntarg)+w/(dtarg*dtarg)))
		Cfracs.append(np.sqrt((1./ntarg)+w/(dtarg*dtarg)))
		i += 1
	return derrs, Cerrs, Cfracs


def add(nu,incti,xs):
	n.append(float(nu))
	inct.append(float(incti))
	x.append(float(xs))

def createC(nu,incti,disti):
	i = 0
	C = []
	while i < len(n):
		C.append((nu[i]-incti[i]*cor)*disti[i]*disti[i]/incti[i])
		i += 1
	return C

def clean():
	i = 0
	while i < len(C):
		if d[i] < 0.1:
			d.remove(d[i])
			C.remove(C[i])
			Cfracs.remove(Cfracs[i])
			Cerrs.remove(Cerrs[i])
			derrs.remove(derrs[i])
			n.remove(n[i])
			inct.remove(inct[i])
			x.remove(x[i])
			i = 0
		else:
			i += 1


def linfit(x,y):
	slope, intercept, r, prob2, see = sps.linregress(x, y)
	mx = np.mean(x)
	sx2 = ((x-mx)**2).sum()
	sd_intercept = see * np.sqrt(1./len(x) + mx*mx/sx2)
	return 	slope, intercept, r, prob2, see, sd_intercept

def avgerrcalc(l):
	s = 0.
	for i in l:
		s = s + i*i
	return np.sqrt(s)/(np.sqrt(len(l)))



def averages(d,C,derrs,Cerrs):
	i = 0
	while i < len(d):
		if d.count(d[i]) != 1:
			dn = np.array(d)
			targinds = np.where(dn == d[i])[0]
			Ctomean = []
			dtomean = []
			derrstomean = []
			Cerrstomean = []
			for ind in targinds:
				Ctomean.append(C[ind])
				dtomean.append(d[ind])
				derrstomean.append(derrs[ind])
				Cerrstomean.append(Cerrs[ind])
			for t in Ctomean:
				C.remove(t)
			for t in dtomean:
				d.remove(t)
			for t in derrstomean:
				derrs.remove(t)
			for t in Cerrstomean:
				Cerrs.remove(t)
			Cmean = np.mean(Ctomean)
			dmean = np.mean(dtomean)
			derrmean = avgerrcalc(derrstomean)
			Cerrmean = avgerrcalc(Cerrstomean)
			#derrmean = np.std(dtomean)
			#derrmean = derrstomean[0]
			#Cerrmean = np.std(Ctomean)
			d.append(dmean)
			C.append(Cmean)
			derrs.append(derrmean)
			Cerrs.append(Cerrmean)
			i = 0
		else:
			i += 1
	return d, C, derrs, Cerrs

add(36169, 1, 335)
add(108555, 3, 335)
add(14241, 3, 375)
add(28385, 6, 375)
add(51511, 11, 375)
add(14621, 11, 425)
add(7836, 6, 425)
add(1316, 1, 425)
add(622,1, 475)
add(3500, 6, 475)
add(3568,6, 475)
add(3411, 6, 475)
add(601, 1, 475)
add(1936, 6, 525)
add(1969, 6, 525)
add(2609, 8, 525)
add(800, 6, 625)
add(1079, 8, 625)
add(1387, 10, 625)
add(661, 8, 725)
add(789, 10, 725)
add(915, 12, 725)
add(1528, 20, 725)
add(945, 20, 825)
add(1469, 30, 825)
add(1001, 30, 925)
add(1201, 40, 925)
add(1562, 50, 925)
add(13969, 1, 350)
add(17983, 1, 345)
add(10915, 1, 355)
add(1434, 10, 625)
add(1985, 10, 575)
add(1592, 8, 575)
add(1235, 12, 675)
add(2012,20,675)
add(1494,60,1000)
add(793, 70, 1235)
add(907, 80, 1235)
add(1149, 100, 1235)
add(1223,80,1125)



d = [(i - (x0-3.))*0.001 for i in x]
C = createC(n,inct,d)
num = ntrials(n)
derrs, Cerrs, Cfracs = builderrs(num, n, d, C)

#print ('Derrs' + str(derrs))
#print ('Cerrs' + str(Cerrs))
#print('Cfracs' + str(Cfracs))
#print(C[-1])

plt.figure(1)
#plt.errorbar(d, C, xerr = derr, yerr =Cerrs, fmt='o')
d,C,derrs,Cerrs = averages(d,C,derrs,Cerrs)
plt.errorbar(d, C, xerr = derr, yerr =Cerrs, fmt='o')
clean()
print(linfit(d,C))
slope, intercept, r_value, p_value, std_err, sd_intercept = linfit(d, C)
print('intercept = ' + str(intercept))
print('error on intercept = ' + str(sd_intercept))
#plt.plot(np.arange(0,1.0,0.001), [x*slope + intercept for x in np.arange(0,1.0,0.001)])

plt.xlabel('Separation/m')
plt.ylabel('Distance Adjusted Count/ (m**2)/s')
plt.show(1)



plt.figure(2)
plt.errorbar(d, C, xerr = derr, yerr =Cerrs, fmt='o')
slope, intercept, r_value, p_value, std_err, sd_intercept = linfit(d, C)
plt.plot(np.arange(0,1.0,0.001), [x*slope + intercept for x in np.arange(0,1.0,0.001)])
plt.xlabel('Separation/m')
plt.ylabel('Distance Adjusted Count/ (m**2)/s')
plt.show(2)


