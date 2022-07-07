import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

plt.close("all")

## VF
def viewfactor(theta1,A2,theta2,d):
	# Calcs view factor from square surfaces 1 to 2
	vf = np.cos(theta1)*np.cos(theta2)*A2/(np.pi*d**2)
	return vf

r1 = 1/2
r2 = 1
t1 = 0
t2 = 0
L = 1

vf1 = viewfactor(t1,np.pi*r1**2,t2,L)
print(vf1)
rri = r1/L
rrj = r2/L
S = (1+rrj**2)/(rri**2)
vf2 = 0.5*(S - (S**2 - 4*(r1/r2)**2)**0.5)
print(vf2)


## DATA
iono_dat = pd.read_csv('Ionosphere Data _ Final Exam.csv')
iono_dat = iono_dat.iloc[:,0:5]
print(iono_dat)

plt.plot(iono_dat.iloc[:,0],iono_dat.iloc[:,1])
#plot1 = iono_dat.plot()
#plt.grid()
plt.show()
print(iono_dat.iloc[:,0:1])

'''
print(iono_dat.to_string()) 
print(iono_dat.iloc[1,1])
print(iono_dat.iloc[0,0])
print(iono_dat.iloc[1,0])
print(iono_dat.iloc[:,0:5])
#print(type(iono_dat))
'''
