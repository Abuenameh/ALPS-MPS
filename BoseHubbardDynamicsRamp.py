__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time
from copy import deepcopy
from scipy import sparse
import datetime
import os
import matplotlib.pyplot as plt
import pyalps.plot

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

W_i = 3e11#7.9e10
W_f = 1e11#1.1e12

def JW(W):
    lenW = len(W)
    J = np.zeros(lenW-1)
    for i in range(0, lenW-1):
        J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng ** 2 + W[i] ** 2) * np.sqrt(Ng ** 2 + W[i+1] ** 2))
    return J


def UW(W):
    return -2*(g24 ** 2) / Delta * (Ng ** 2 * W ** 2) / ((Ng ** 2 + W ** 2) ** 2)

def Wt(W_i, W_f, tau, t):
    if t < tau:
        return W_i + (W_f - W_i) * t / tau
    else:
        return W_f + (W_i - W_f) * (t - tau) / tau

def ramp(W_i, W_f, xi, tau, dt):
    return [Wt(W_i, W_f, tau, i*dt)*xi for i in 1+np.arange(2*tau / dt)]

L = 10
nmax = 5
sweeps = 200
maxstates = 200

tf = 1e-6
numsteps = 1000
dt = 0.5e-10#1e-10#tf / numsteps

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
# parms['MEASURE_LOCAL[Local density]'] = 'n'
# parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
# parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'
# parms['MEASURE_CORRELATIONS[Density density]'] = 'n:n'
# parms['always_measure'] = 'Local density,Local density squared,One body density matrix,Density density'
parms['DT'] = dt
# parms['TIMESTEPS'] = numsteps
parms['COMPLEX'] = 1
parms['N_total'] = L
parms['init_state'] = 'local_quantumnumbers'
parms['initial_local_N'] = ','.join(['1']*L)
parms['te_order'] = 'second'
parms['update_each'] = 1
xi = (1 + 0.5 * np.random.uniform(-1, 1, L))
for i in range(L-1):
    parms['t'+str(i)] = mathematica(JW(W_i*xi)[i])
for i in range(L):
    parms['U'+str(i)] = mathematica(UW(W_i*xi)[i])

resi = 302
basename = 'DynamicsTasks/bhramp.'+str(L)+'.'+str(resi)
# gbasename = basename
gbasename = 'DynamicsTasks/bhramp.'+str(L)+'.300'

start = datetime.datetime.now()

input_file = pyalps.writeInputFiles(gbasename+'.ground',[parms])
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

initstate = pyalps.getResultFiles(prefix=gbasename+'.ground')[0].replace('xml', 'chkp')

parms['initfile'] = initstate
parms['MEASURE_OVERLAP[Overlap]'] = initstate
parms['always_measure'] = 'Overlap,Local density,Local density squared,One body density matrix,Density density'

taus = [1e-6]#np.linspace(1e-7, 2e-7, 2)#[1e-7,1.1e-7,1.2e-7]

parmslist = []
for tau in taus:
    parmsi = deepcopy(parms)
    parmsi['tau'] = tau
    steps = round(2*tau / dt)
    parmsi['TIMESTEPS'] = steps#int(2*tau / dt)
    parmsi['DT'] = 2*tau / steps
    parmsi['measure_each'] = steps#int(2*tau / dt)
    Wt = ramp(W_i, W_f, xi, tau, dt)
    Jt = [JW(W) for W in Wt]
    Ut = [UW(W) for W in Wt]
    for i in range(L-1):
        parmsi['t'+str(i)+'[Time]'] = ','.join([mathematica(J[i]) for J in Jt])
        # parmsi['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)[i]) for W in ramp(W_i, W_f, xi, tau, dt)])
    for i in range(L):
        parmsi['U'+str(i)+'[Time]'] = ','.join([mathematica(U[i]) for U in Ut])
        # parmsi['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)[i]) for W in ramp(W_i, W_f, xi, tau, dt)])
    parmslist.append(parmsi)


input_file = pyalps.writeInputFiles(basename+'.dynamic',parmslist)
res = pyalps.runApplication('mps_evolve',input_file,writexml=True)

end = datetime.datetime.now()

## simulation results
# data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename+'.dynamic'), what=['Overlap', 'Local density', 'Local density squared', 'One body density matrix', 'Density density'])
data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename+'.dynamic'), what=['Overlap', 'Energy'])

coords = []
for d1 in data:
    for s1 in d1:
        for d in s1:
            for s in d:
                if(s.props['observable'] == 'One body density matrix'):
                    coords = (s.x[:,0], s.x[:,1])

XY = pyalps.collectXY(data, x='Time', y='Overlap', foreach=['tau'])
p = [[[(x + 1)*dt, 1-abs(y**2)] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='Energy', foreach=['tau'])
E = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='Local density', foreach=['tau'])
n = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='Local density squared', foreach=['tau'])
n2 = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='One body density matrix', foreach=['tau'])
corr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='Density density', foreach=['tau'])
ncorr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

resultsfile = open(os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/rampres.'+str(resi)+'.txt'), 'w')
resultsstr = ''
resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
resultsstr += 'numsteps['+str(resi)+']='+str(numsteps)+';\n'
resultsstr += 'dt['+str(resi)+']='+mathematica(dt)+';\n'
resultsstr += 'tf['+str(resi)+']='+mathematica(tf)+';\n'
resultsstr += 'p['+str(resi)+']='+mathematica(p)+';\n'
resultsstr += 'En['+str(resi)+']='+mathematica(E)+';\n'
resultsstr += 'n['+str(resi)+']='+mathematica(n)+';\n'
resultsstr += 'n2['+str(resi)+']='+mathematica(n2)+';\n'
resultsstr += 'corr['+str(resi)+']='+mathematica(corr)+';\n'
resultsstr += 'ncorr['+str(resi)+']='+mathematica(ncorr)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)

