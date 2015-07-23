__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time
from copy import deepcopy
from scipy import sparse
import datetime
import os

def mathematica(x):
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        try:
            return '{:.20f}'.format(x).replace('j', 'I')
        except:
            return str(x)

Na = 1000
g13 = 2.5e9
g24 = 2.5e9
Delta = -2.0e10
alpha = 1.1e7

Ng = np.sqrt(Na) * g13

def JW(W):
    return alpha * W ** 2 / (np.sqrt(Ng ** 2 + W ** 2) ** 2)
    # lenW = len(W)
    # J = np.zeros(lenW)
    # for i in range(0, lenW-1):
    #     J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng * Ng + W[i] * W[i]) * np.sqrt(Ng * Ng + W[i+1] * W[i+1]))
    # J[lenW-1] = alpha * W[lenW-1] * W[0] / (np.sqrt(Ng * Ng + W[lenW-1] * W[lenW-1]) * np.sqrt(Ng * Ng + W[0] * W[0]))
    # return J


def UW(W):
    return -2*(g24 ** 2) / Delta * (Ng ** 2 * W ** 2) / ((Ng ** 2 + W ** 2) ** 2)

def func(x):
    if x < 0:
        return 0
    elif x < 0.5:
        return 2 * x ** 2
    elif x < 1:
        return -2 * (x - 1) ** 2 + 1
    else:
        return 1

def Wt(W_i, W_f, t):
    return (W_i - W_f) * func(2 * (1 - 1e6*t) - 0.4) + W_f

# \[CapitalOmega]i := 7.9*^10
# \[CapitalOmega]f := 1.1*^12

def quench(W_i, W_f, steps, dt):
    return [Wt(W_i, W_f, i*dt) for i in 1+np.arange(steps)]

# print mathematica(quench(7.9e10, 1.1e12, 100, 1e-6/100))
# quit()

basename = 'Tasks/bh'+str(time.time())

L = 10
nmax = 5
sweeps = 200
maxstates = 400

tf = 1e-6
numsteps = 1000

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice'+str(L)+'.xml'
parms['LATTICE'] = 'inhomogeneous open chain lattice'
# parms['LATTICE'] = 'open chain lattice'
parms['MODEL_LIBRARY'] = 'model.xml'
parms['MODEL'] = 'boson Hubbard'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = nmax
parms['SWEEPS'] = sweeps
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = maxstates
parms['MEASURE_LOCAL[Local density]'] = 'n'
parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'
parms['MEASURE_CORRELATIONS[Density density]'] = 'n:n'
parms['always_measure'] = 'Local density,Local density squared,One body density matrix,Density density'
parms['init_state'] = 'local_quantumnumbers'
parms['DT'] = tf / numsteps
parms['TIMESTEPS'] = numsteps
parms['COMPLEX'] = 1
parms['N_total'] = L
parms['init_state'] = 'local_quantumnumbers'
parms['initial_local_N'] = ','.join(['1']*L)
parms['te_order'] = 'second'
parms['update_each'] = 1
for i in range(L-1):
    parms['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)) for W in quench(7.9e10, 1.1e12, numsteps, tf / numsteps)])
for i in range(L):
    parms['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)) for W in quench(7.9e10, 1.1e12, numsteps, tf / numsteps)])

resi = 19
basename = 'DynamicsTasks/bhd.'+str(L)+'.'+str(resi)

input_file = pyalps.writeInputFiles(basename,[parms])
res = pyalps.runApplication('mps_evolve',input_file,writexml=True)

data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename), what=['Local density', 'Local density squared', 'One body density matrix', 'Density density'])

nt = pyalps.collectXY(data, x='Time', y='Local density', ignoreProperties=True)[0].y
n2t = pyalps.collectXY(data, x='Time', y='Local density squared', ignoreProperties=True)[0].y
corrt = pyalps.collectXY(data, x='Time', y='One body density matrix', ignoreProperties=True)[0].y
ncorrt = pyalps.collectXY(data, x='Time', y='Density density', ignoreProperties=True)[0].y

Ft = n2t - nt**2
print mathematica(Ft[:,5])

# print(mathematica(ncorrt))

resultsfile = open(os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/dres.'+str(resi)+'.txt'), 'w')
resultsstr = ''
resultsstr += 'nt['+str(resi)+']='+mathematica(nt)+';\n'
resultsstr += 'n2t['+str(resi)+']='+mathematica(n2t)+';\n'
resultsstr += 'corrt['+str(resi)+']='+mathematica(corrt)+';\n'
resultsstr += 'ncorrt['+str(resi)+']='+mathematica(ncorrt)+';\n'
resultsstr += 'Ft['+str(resi)+']='+mathematica(Ft)+';\n'
resultsfile.write(resultsstr)
