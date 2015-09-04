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
    return 'rampres.' + str(i) + '.txt'

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
    # J[lenW-1] = alpha * W[lenW-1] * W[0] / (np.sqrt(Ng ** 2 + W[lenW-1] ** 2) * np.sqrt(Ng ** 2 + W[0] ** 2))
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
    return [Wt(W_i, W_f, i*dt)*xi for i in 1+np.arange(tf / dt)]

# print mathematica(quench(7.9e10, 1.1e12, 100, 1e-6/100))
# quit()

basename = 'Tasks/bh'+str(time.time())

L = 10
nmax = 5
sweeps = 400
maxstates = 400

tf = 1e-6
dt = float(sys.argv[3])#5e-10#1e-10#tf / numsteps
numsteps = int(tf / dt)

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
# parms['always_measure'] = 'Local density,Local density squared,One body density matrix,Density density'
parms['init_state'] = 'local_quantumnumbers'
parms['DT'] = dt
# parms['TIMESTEPS'] = numsteps
parms['COMPLEX'] = 1
parms['N_total'] = L
parms['init_state'] = 'local_quantumnumbers'
parms['initial_local_N'] = ','.join(['1']*L)
parms['te_order'] = 'second'
parms['update_each'] = 1
np.random.seed(int(sys.argv[1]))
xi = (1 + 0.5 * np.random.uniform(-1, 1, L))
for i in range(L-1):
    parms['t'+str(i)] = mathematica(JW(W_i*xi)[i])#','.join([mathematica(JW(W_i))])
    # parms['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)) for W in quench(W_i, W_f, numsteps, tf / numsteps)])
for i in range(L):
    parms['U'+str(i)] = mathematica(UW(W_i*xi)[i])#','.join([mathematica(UW(W_i))])
    # parms['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)) for W in quench(W_i, W_f, numsteps, tf / numsteps)])

resi = int(sys.argv[2])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

# resi = 9
basename = 'DynamicsTasks/bhramp.'+str(L)+'.'+str(resi)

start = datetime.datetime.now()

parms['always_measure'] = 'Local density,Local density squared,One body density matrix,Density density'
parms['measure_each'] = 1#numsteps

taus = np.linspace(1e-7, 2e-7, 2)#[1e-7,1.1e-7,1.2e-7]

# parmslist = []
parms['tau'] = 1
parms['TIMESTEPS'] = int(tf / dt)
for i in range(L-1):
    parms['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)[i]) for W in quench(W_i, W_f, xi, tf, dt)])
for i in range(L):
    parms['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)[i]) for W in quench(W_i, W_f, xi, tf, dt)])
parmslist = [parms]
# for tau in taus:
#     parmsi = deepcopy(parms)
#     parmsi['tau'] = tau
#     parmsi['TIMESTEPS'] = int(2*tau / dt)
#     for i in range(L-1):
#         parmsi['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)[i]) for W in quench(W_i, W_f, xi, 2*tau, dt)])
#     for i in range(L):
#         parmsi['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)[i]) for W in quench(W_i, W_f, xi, 2*tau, dt)])
#     parmslist.append(parmsi)


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

# XY = pyalps.collectXY(data, x='Time', y='Overlap', foreach=['tau'])
# p = [[[(x + 1)*dt, 1-abs(y**2)] for (x, y) in zip(xy.x, xy.y)] for xy in XY]

XY = pyalps.collectXY(data, x='Time', y='Local density', foreach=['tau'])
n = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='Local density squared', foreach=['tau'])
n2 = [[[(x + 1)*dt, y] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='One body density matrix', foreach=['tau'])
corr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

XY = pyalps.collectXY(data, x='Time', y='Density density', foreach=['tau'])
ncorr = [[[(x + 1)*dt, sparse.coo_matrix((y, coords)).toarray()] for (x, y) in zip(xy.x, xy.y)] for xy in XY][0]

# print E

# t = []
# nt = []
# for d1 in data:
#     for s1 in d1:
#         for d in s1:
#             for s in d:
#                 if(s.props['observable'] == 'Local density'):
#                     t += [s.props['Time']]
#                     nt += [s.y[0]]
#
# tord = np.argsort(t)
# t = np.array(t)[tord]
# nt = np.array(nt)[tord]
#
# print nt[-1]

# print F
# plt.figure()
# pyalps.plot.plot(F)
# plt.xlabel('Time $t$')
# plt.ylabel('Loschmidt Echo $|< \psi(0)|\psi(t) > |^2$')
# plt.title('Loschmidt Echo vs. Time')
# plt.legend(loc='lower right')
#
# plt.show()

resultsfile = open(os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/rampres.'+str(resi)+'.txt'), 'w')
resultsstr = ''
resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
resultsstr += 'numsteps['+str(resi)+']='+str(numsteps)+';\n'
resultsstr += 'dt['+str(resi)+']='+mathematica(dt)+';\n'
resultsstr += 'tf['+str(resi)+']='+mathematica(tf)+';\n'
resultsstr += 'xi['+str(resi)+']='+mathematica(xi)+';\n'
# resultsstr += 'p['+str(resi)+']='+mathematica(p)+';\n'
resultsstr += 'En['+str(resi)+']='+mathematica(E)+';\n'
resultsstr += 'n['+str(resi)+']='+mathematica(n)+';\n'
resultsstr += 'n2['+str(resi)+']='+mathematica(n2)+';\n'
resultsstr += 'corr['+str(resi)+']='+mathematica(corr)+';\n'
resultsstr += 'ncorr['+str(resi)+']='+mathematica(ncorr)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)

print 'Res: ' + str(resi)
print str(end-start)

quit()

data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename), what=['Local density', 'Local density squared', 'One body density matrix', 'Density density'])
# print data[0][0][0]
# quit()

t = []
nt = []
n2t = []
corrt = []
ncorrt = []
for d1 in data:
    for s1 in d1:
        for d in s1:
            for s in d:
                if(s.props['observable'] == 'Local density'):
                    t += [s.props['Time']]
                    nt += [s.y[0]]
                    # nresults += [(s.props['N_total'], s.y[0])]
                if(s.props['observable'] == 'Local density squared'):
                    n2t += [s.y[0]]
                    # n2results += [(s.props['N_total'], s.y[0])]
                if(s.props['observable'] == 'One body density matrix'):
                    corrt += [sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()]
                    # corrresults += [(s.props['N_total'], sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray())]
                if(s.props['observable'] == 'Density density'):
                    ncorrt += [sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()]
                    # ncorrresults += [(s.props['N_total'], sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray())]

tord = np.argsort(t)
t = np.array(t)[tord]
nt = np.array(nt)[tord]
n2t = np.array(n2t)[tord]
corrt = np.array(corrt)[tord]
ncorrt = np.array(ncorrt)[tord]

# nt = pyalps.collectXY(data, x='Time', y='Local density', ignoreProperties=True)[0].y
# n2t = pyalps.collectXY(data, x='Time', y='Local density squared', ignoreProperties=True)[0].y
# corrt = pyalps.collectXY(data, x='Time', y='One body density matrix', ignoreProperties=True)[0].y
# ncorrt = pyalps.collectXY(data, x='Time', y='Density density', ignoreProperties=True)[0].y

# print len(data[0][0])
# print(data[0][0][1])

Ft = n2t - nt**2
# print mathematica(Ft[:,5])

# print(mathematica(ncorrt))

resultsfile = open(os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/dres.'+str(resi)+'.txt'), 'w')
resultsstr = ''
# resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
resultsstr += 'numsteps['+str(resi)+']='+str(numsteps)+';\n'
resultsstr += 'dt['+str(resi)+']='+mathematica(dt)+';\n'
resultsstr += 'tf['+str(resi)+']='+mathematica(tf)+';\n'
resultsstr += 'nt['+str(resi)+']='+mathematica(nt)+';\n'
resultsstr += 'n2t['+str(resi)+']='+mathematica(n2t)+';\n'
resultsstr += 'corrt['+str(resi)+']='+mathematica(corrt)+';\n'
resultsstr += 'ncorrt['+str(resi)+']='+mathematica(ncorrt)+';\n'
resultsstr += 'Ft['+str(resi)+']='+mathematica(Ft)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)
