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
import sys

def mathematica(x):
    if x == 'True' or x == 'False':
        return x
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        try:
            return '{:.16}'.format(x).replace('j', 'I').replace('e', '*^')
        except:
            return str(x)

def resifile(i):
    return 'rampplenio.' + str(i) + '.txt'

Na = 1000
g13 = 2.5e9
g24 = 2.5e9
Delta = -2.0e10
alpha = 1.1e7

Ng = np.sqrt(Na) * g13

W_i = 7.9e10
W_f = 1.1e12

def JW(W):
    lenW = len(W)
    J = np.zeros(lenW-1)
    for i in range(0, lenW-1):
        J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng ** 2 + W[i] ** 2) * np.sqrt(Ng ** 2 + W[i+1] ** 2))
    return J


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

def quench(W_i, W_f, xi, tf, dt):
    # return [W_i*xi for i in 1+np.arange(tf / dt)]
    return [Wt(W_i, W_f, i*dt)*xi for i in 1+np.arange(tf / dt)]

L = 10
nmax = 5
maxstates = 400

tf = 1e-6
dt = float(sys.argv[4])#5e-10#1e-10#tf / numsteps
numsteps = int(tf / dt + 0.5)
dt = tf / numsteps

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice'+str(L)+'.xml'
parms['LATTICE'] = 'inhomogeneous open chain lattice'
# parms['LATTICE'] = 'chain lattice'
parms['MODEL_LIBRARY'] = 'model.xml'
parms['MODEL'] = 'boson Hubbard'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = nmax
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = maxstates
parms['MEASURE_LOCAL[Local density]'] = 'n'
parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'
parms['MEASURE_CORRELATIONS[Density density]'] = 'n:n'
parms['init_state'] = 'local_quantumnumbers'
parms['dt'] = dt
parms['TIMESTEPS'] = numsteps
parms['COMPLEX'] = 1
parms['N_total'] = L
parms['init_state'] = 'local_quantumnumbers'
parms['initial_local_N'] = ','.join(['1']*L)
parms['te_order'] = 'second'
parms['update_each'] = 1#10**8#-1#1
parms['chkp_each'] = 10**8#1000
# parms['TIMESTEPS'] = int(tf / dt)
# parms['update_each'] = int(tf / dt)

seed = int(sys.argv[1])
ximax = float(sys.argv[2])
np.random.seed(seed)
xi = (1 + ximax * np.random.uniform(-1, 1, L))

resi = int(sys.argv[3])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

basename = 'DynamicsTasks/rampplenio.'+str(L)+'.'+str(resi)

start = datetime.datetime.now()

parms['always_measure'] = 'Local density,Local density squared,One body density matrix,Density density'
parms['measure_each'] = max(1, numsteps/200)

for i in range(L-1):
    parms['t'+str(i)] = mathematica(JW(W_i*xi)[i])
for i in range(L):
    parms['U'+str(i)] = mathematica(UW(W_i*xi)[i])

parms['tau'] = 1
for i in range(L-1):
    parms['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)[i]) for W in quench(W_i, W_f, xi, tf, dt)])
for i in range(L):
    parms['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)[i]) for W in quench(W_i, W_f, xi, tf, dt)])
parmslist = [parms]

# print mathematica([JW(W)[0] for W in quench(W_i, W_f, xi, tf, dt)])
# print mathematica([UW(W)[0] for W in quench(W_i, W_f, xi, tf, dt)])
# quit()

input_file = pyalps.writeInputFiles(basename+'.dynamic',parmslist)
res = pyalps.runApplication('mps_evolve',input_file,writexml=True)

end = datetime.datetime.now()

## simulation results
data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename+'.dynamic'), what=['Energy', 'Local density', 'Local density squared', 'One body density matrix', 'Density density'])

coords = []
for d1 in data:
    for s1 in d1:
        for d in s1:
            for s in d:
                if(s.props['observable'] == 'One body density matrix'):
                    coords = (s.x[:,0], s.x[:,1])

XY = pyalps.collectXY(data, x='Time', y='Energy', foreach=['tau'])
E = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='Local density', foreach=['tau'])
n = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='Local density squared', foreach=['tau'])
n2 = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='One body density matrix', foreach=['tau'])
corr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='Density density', foreach=['tau'])
ncorr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

resultsfile = open(os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/rampplenio.'+str(resi)+'.txt'), 'w')
resultsstr = ''
resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
resultsstr += 'ximax['+str(resi)+']='+str(ximax)+';\n'
resultsstr += 'numsteps['+str(resi)+']='+str(numsteps)+';\n'
resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
resultsstr += 'numsteps['+str(resi)+']='+str(numsteps)+';\n'
resultsstr += 'dt['+str(resi)+']='+mathematica(dt)+';\n'
resultsstr += 'tf['+str(resi)+']='+mathematica(tf)+';\n'
resultsstr += 'xi['+str(resi)+']='+mathematica(xi)+';\n'
resultsstr += 'En['+str(resi)+']='+mathematica(E)+';\n'
resultsstr += 'n['+str(resi)+']='+mathematica(n)+';\n'
resultsstr += 'n2['+str(resi)+']='+mathematica(n2)+';\n'
resultsstr += 'corr['+str(resi)+']='+mathematica(corr)+';\n'
resultsstr += 'ncorr['+str(resi)+']='+mathematica(ncorr)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)

print 'Res: ' + str(resi)
print str(end-start)

# print parms['U1[Time]']

