__author__ = 'Abuenameh'

import pyalps
import numpy as np
import scipy.optimize
from collections import OrderedDict
import time
from copy import deepcopy
from scipy import sparse
import datetime
import sys
import os
import progressbar
import concurrent.futures
import random
from Numberjack import VarArray, Model, Sum, Minimize
import subprocess

def mathematica(x):
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        try:
            return '{:.16}'.format(x).replace('j', 'I').replace('e', '*^')
        except:
            return str(x)

def resifile(i):
    return 'res.' + str(i) + '.txt'

def makeres(n, m):
    return np.zeros((n, m)).tolist()

periodic = int(sys.argv[4]) == 1
twist = int(sys.argv[5]) == 1

numthreads = 35

L = 25
nmax = 5
sweeps = 200
maxstates = 200

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice' + str(L) + '.xml'
if periodic:
    parms['LATTICE'] = 'inhomogeneous periodic chain lattice'
else:
    parms['LATTICE'] = 'inhomogeneous open chain lattice'
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
parms['init_state'] = 'local_quantumnumbers'

bounds = tuple([(0, nmax)] * L)

seed = int(sys.argv[1])
np.random.seed(seed)
ximax = float(sys.argv[2])
xi = (1 + ximax * np.random.uniform(-1, 1, L))
# xi = np.array([1.0488135039273247529, 1.2151893663724195882, 1.1027633760716439859, \
# 1.04488318299689675328, 0.9236547993389047084])
# xi = np.array([1]*L)
xisort = sorted([(xii, i) for (i, xii) in enumerate(xi)])

resi = int(sys.argv[3])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

basename = 'Tasks/bh.' + str(L) + '.' + str(resi) + '.'

def runmps(task, it, iN, Ui, ti, N):
    parmsi = deepcopy(parms)
    parmsi['it'] = it
    parmsi['iN'] = iN

    t = xi * ti
    if twist:
        t[0] *= -1
    U = xi * Ui
    for i in range(L):
        parmsi['t'+str(i)] = t[i]
    for i in range(L):
        parmsi['U'+str(i)] = U[i]

    parmsi['N_total'] = N

    try:
        if ximax == 0:
            raise ValueError
        ns = VarArray(L, nmax)
        E = Sum([n*(n-1) for n in ns], U.tolist())
        model = Model(Minimize(E), [Sum(ns) == N])
        solver = model.load('SCIP')
        solver.solve()
    except:
        basen = N // L
        ns = [basen] * L
        rem = N % L
        excessi = [i for (xii, i) in xisort[:rem]]
        for i in excessi:
            ns[i] += 1
    parmsi['initial_local_N'] = ','.join([str(n) for n in ns])

    input_file = pyalps.writeInputFiles(basename + str(task), [parmsi])
    # subprocess.call(['bash','-c','read'])
    pyalps.runApplication('mps_optim', input_file, writexml=True)

def main():
    ts = [0.01]#[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]#[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]#[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]#[0.01,0.1]#np.linspace(0.01, 0.05, 5).tolist()
    nt = len(ts)
    Us = [1]*nt
    Ns = range(0, 2*L+1)#range(23,27)#range(25,2*L+1)#[35,36,37]#[32]*12#range(32,40)#range(38, 46)#[40,41,42,43]#range(25, 2*L+1)#range(51,70)#[66,66,66,66,66,66,66,66,66]#[66,67,68]#[66,67,68,69,70]#range(0, 2*L+1)
    nN = len(Ns)
    tUNs = zip(range(nt*nN), [[i, j] for i in range(nt) for j in range(nN)], [[Ui, ti, Ni] for (Ui, ti) in zip(Us, ts) for Ni in Ns])
    ntasks = len(tUNs)

    start = datetime.datetime.now()

    pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.Timer()], maxval=ntasks).start()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmps, task, it, iN, Ui, ti, N) for (task, [it, iN], [Ui, ti, N]) in tUNs]
        for future in pbar(concurrent.futures.as_completed(futures)):
            future.result()

    end = datetime.datetime.now()

    #load all measurements for all states
    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

    Es = makeres(nt, nN)
    ns = makeres(nt, nN)
    n2s = makeres(nt, nN)
    corrs = makeres(nt, nN)
    ncorrs = makeres(nt, nN)
    for d in data:
        for s in d:
            it = int(s.props['it'])
            iN = int(s.props['iN'])
            if(s.props['observable'] == 'Energy'):
                Es[it][iN] = s.y[0]
            if(s.props['observable'] == 'Local density'):
                ns[it][iN] = s.y[0]
            if(s.props['observable'] == 'Local density squared'):
                n2s[it][iN] = s.y[0]
            if(s.props['observable'] == 'One body density matrix'):
                corrs[it][iN] = sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()
            if(s.props['observable'] == 'Density density'):
                ncorrs[it][iN] = sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()

    resultsfile = open(resdir + 'res.'+str(resi)+'.txt', 'w')
    resultsstr = ''
    resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
    resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
    resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
    resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
    resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
    resultsstr += 'periodic['+str(resi)+']='+str(periodic)+';\n'
    resultsstr += 'twisted['+str(resi)+']='+str(twist)+';\n'
    resultsstr += 'xi['+str(resi)+']='+mathematica(xi)+';\n'
    resultsstr += 'ts['+str(resi)+']='+mathematica(ts)+';\n'
    resultsstr += 'Us['+str(resi)+']='+mathematica(Us)+';\n'
    resultsstr += 'Ns['+str(resi)+']='+mathematica(Ns)+';\n'
    resultsstr += 'Eres['+str(resi)+']='+mathematica(Es)+';\n'
    resultsstr += 'nres['+str(resi)+']='+mathematica(ns)+';\n'
    resultsstr += 'n2res['+str(resi)+']='+mathematica(n2s)+';\n'
    resultsstr += 'corrres['+str(resi)+']='+mathematica(corrs)+';\n'
    resultsstr += 'ncorrres['+str(resi)+']='+mathematica(ncorrs)+';\n'
    resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
    resultsfile.write(resultsstr)

    print 'Res: ' + str(resi)

if __name__ == '__main__':
    main()