import matplotlib.pyplot as plt
import scipy as sci
import numpy as np


x0 = 253  # in mm

x0err = 0.001
xerr = 0.001
derr = np.sqrt(x0err*x0err + xerr*xerr)*0.001
cor = 0.3

n = []
inct = []
x = []


def ntrials(n):
	return len(n)

def builderrs(num,ns,ds,Cs):
	w = 4. * 2. * 0.000001
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



add(77551,3,263)
add(4773, 3, 353)
add(176, 3, 753)









d = [(i - (x0-3.))*0.001 for i in x]
C = createC(n,inct,d)
num = ntrials(n)
derrs, Cerrs, Cfracs = builderrs(num, n, d, C)

print(Cfracs)


plt.figure()
plt.errorbar(d, C, xerr = derr, yerr =Cerrs)
plt.show()

