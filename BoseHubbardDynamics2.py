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

W_i = 7.9e10
W_f = 1.1e12

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

def quench(W_i, W_f, tf, dt):
    return [Wt(W_i, W_f, i*dt) for i in 1+np.arange(tf / dt)]

# print mathematica(quench(7.9e10, 1.1e12, 100, 1e-6/100))
# quit()

basename = 'Tasks/bh'+str(time.time())

L = 10
nmax = 5
sweeps = 200
maxstates = 200

tf = 1e-6
numsteps = 1000
dt = 1e-10#tf / numsteps

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
parms['init_state'] = 'local_quantumnumbers'
parms['DT'] = dt
# parms['TIMESTEPS'] = numsteps
parms['COMPLEX'] = 1
parms['N_total'] = L
parms['init_state'] = 'local_quantumnumbers'
parms['initial_local_N'] = ','.join(['1']*L)
parms['te_order'] = 'second'
parms['update_each'] = 1
for i in range(L-1):
    parms['t'+str(i)] = ','.join([mathematica(JW(W_i))])
    # parms['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)) for W in quench(W_i, W_f, numsteps, tf / numsteps)])
for i in range(L):
    parms['U'+str(i)] = ','.join([mathematica(UW(W_i))])
    # parms['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)) for W in quench(W_i, W_f, numsteps, tf / numsteps)])

resi = 2
basename = 'DynamicsTasks/bhramp.'+str(L)+'.'+str(resi)

start = datetime.datetime.now()

input_file = pyalps.writeInputFiles(basename+'.ground',[parms])
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

initstate = pyalps.getResultFiles(prefix=basename+'.ground')[0].replace('xml', 'chkp')

parms['initfile'] = initstate
parms['MEASURE_OVERLAP[Overlap]'] = initstate
parms['always_measure'] = 'Overlap'

taus = np.linspace(1e-7, 1e-6, 100)#[1e-7,1.1e-7,1.2e-7]

parmslist = []
for tau in taus:
    parmsi = deepcopy(parms)
    parmsi['tau'] = tau
    parmsi['TIMESTEPS'] = int(2*tau / dt)
    for i in range(L-1):
        parmsi['t'+str(i)+'[Time]'] = ','.join([mathematica(JW(W)) for W in quench(W_i, W_f, 2*tau, dt)])
    for i in range(L):
        parmsi['U'+str(i)+'[Time]'] = ','.join([mathematica(UW(W)) for W in quench(W_i, W_f, 2*tau, dt)])
    parmslist.append(parmsi)


input_file = pyalps.writeInputFiles(basename+'.dynamic',parmslist)
res = pyalps.runApplication('mps_evolve',input_file,writexml=True)

end = datetime.datetime.now()

## simulation results
data = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename+'.dynamic'), what=['Overlap'])

p = []
F = pyalps.collectXY(data, x='Time', y='Overlap', foreach=['tau'])
for d in pyalps.flatten(F):
    p.append([(d.x[-1] + 1) * d.props['dt'], 1 - abs(d.y[-1])**2])
    # d.x =  (d.x + 1.) * d.props['dt'] # convert time index to real time
    # d.y = abs(d.y)**2 # Loschmidt Echo defined as the module squared of the overlap
    # d.props['label']=r'$\tau={0}$'.format( d.props['tau'] )

print p

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
resultsstr += 'p['+str(resi)+']='+mathematica(p)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)

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
