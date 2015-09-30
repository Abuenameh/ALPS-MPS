__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from scipy import sparse
import datetime
import sys
import os
import progressbar
import concurrent.futures
from Numberjack import VarArray, Model, Sum, Minimize

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
    return 'bhpres.' + str(i) + '.txt'

def makeres(n, m):
    return np.zeros((n, m)).tolist()

Na = 1000
g13 = 2.5e9
g24 = 2.5e9
Delta = -2.0e10
alpha = 1.1e7

Ng = np.sqrt(Na) * g13

def JW(W):
    J = np.zeros(L)
    J[L-1] = alpha * W[L-1] * W[0] / (np.sqrt(Ng ** 2 + W[L-1] ** 2) * np.sqrt(Ng ** 2 + W[0] ** 2))
    # J = np.zeros(L-1)
    for i in range(0, L-1):
        J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng ** 2 + W[i] ** 2) * np.sqrt(Ng ** 2 + W[i+1] ** 2))
    return J

def JWi(W):
    return alpha * W ** 2 / (Ng ** 2 + W ** 2)

def UW(W):
    return -2*(g24 ** 2) / Delta * (Ng ** 2 * W ** 2) / ((Ng ** 2 + W ** 2) ** 2)

numthreads = 35

L = 50
nmax = 5
sweeps = 200
maxstates = 200

#prepare the input parameters
parms = OrderedDict()
# parms['LATTICE_LIBRARY'] = 'lattice' + str(L) + '.xml'
# parms['LATTICE'] = 'inhomogeneous open chain lattice'
parms['LATTICE'] = 'open chain lattice'
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
# parms['init_state'] = 'local_quantumnumbers'
parms['chkp_each'] = sweeps
parms['optimization'] = 'singlesite'
# parms['ngrowsweeps'] = 4
# parms['nmainsweeps'] = 4
# parms['alpha_initial'] = 1e-0
# parms['alpha_main'] = 1e-2
# parms['alpha_final'] = 1e-4

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

basename = 'Tasks/bhp.' + str(L) + '.' + str(resi) + '.'

def runmps(task, iW, iN, Wi, N):
    parmsi = deepcopy(parms)
    parmsi['iW'] = iW
    parmsi['iN'] = iN

    W = Wi * xi
    t = JW(W)
    U = UW(W)
    for i in range(L):
        parmsi['t'+str(i)] = t[i]
    for i in range(L):
        parmsi['U'+str(i)] = U[i]

    parmsi['N_total'] = N

    # try:
    #     if ximax == 0:
    #         raise ValueError
    #     ns = VarArray(L, nmax)
    #     E = Sum([n*(n-1) for n in ns], (0.5*U).tolist())
    #     model = Model(Minimize(E), [Sum(ns) == N])
    #     solver = model.load('SCIP')
    #     solver.setTimeLimit(60)
    #     solved = solver.solve()
    #     parmsi['solved'] = solved
    #     # print >>sys.stderr, str(ns)
    #     # print >>sys.stderr, np.sum(np.multiply([0.5*int(str(n))*(int(str(n))-1) for n in ns], U))
    # except:
    #     basen = N // L
    #     ns = [basen] * L
    #     rem = N % L
    #     excessi = [i for (xii, i) in xisort[:rem]]
    #     for i in excessi:
    #         ns[i] += 1
    # basen = N // L
    # ns2 = [basen] * L
    # rem = N % L
    # excessi = [i for (xii, i) in xisort[:rem]]
    # for i in excessi:
    #     ns2[i] += 1
    # print >>sys.stderr, str(ns2)
    # print >>sys.stderr, np.sum(np.multiply([n*(n-1) for n in ns2], U))
    # parmsi['initial_local_N'] = ','.join([str(n) for n in ns])

    input_file = pyalps.writeInputFiles(basename + str(task), [parmsi])
    pyalps.runApplication('mps_optim', input_file, writexml=True)

def main():
    Ws = [1.5e11]#[7.9e10]#np.linspace(2e11,3.2e11,10)#[2e10]
    nW = len(Ws)
    Ns = range(40,70)#range(0,2*L+1)#range(24,2*L+1)#range(0,2*L+1)#range(23,27)
    nN = len(Ns)
    WNs = zip(range(nW*nN), [[i, j] for i in range(nW) for j in range(nN)], [[Wi, Ni] for Wi in Ws for Ni in Ns])
    ntasks = len(WNs)

    start = datetime.datetime.now()

    pbar = progressbar.ProgressBar(widgets=['Res: '+str(resi)+' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.Timer()], maxval=ntasks).start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmps, task, iW, iN, Wi, N) for (task, [iW, iN], [Wi, N]) in WNs]
        for future in pbar(concurrent.futures.as_completed(futures)):
            future.result()
            sys.stderr.flush()

    end = datetime.datetime.now()

    #load all measurements for all states
    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

    solved = makeres(nW, nN)
    Es = makeres(nW, nN)
    ns = makeres(nW, nN)
    n2s = makeres(nW, nN)
    corrs = makeres(nW, nN)
    ncorrs = makeres(nW, nN)
    for d in data:
        for s in d:
            iW = int(s.props['iW'])
            iN = int(s.props['iN'])
            # solved[iW][iN] = s.props['solved']
            if(s.props['observable'] == 'Energy'):
                Es[iW][iN] = s.y[0]
                # print >>sys.stderr, Es[iW][iN]
            if(s.props['observable'] == 'Local density'):
                ns[iW][iN] = s.y[0]
                # print >>sys.stderr, ns[iW][iN]
            if(s.props['observable'] == 'Local density squared'):
                n2s[iW][iN] = s.y[0]
            if(s.props['observable'] == 'One body density matrix'):
                corrs[iW][iN] = sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()
            if(s.props['observable'] == 'Density density'):
                ncorrs[iW][iN] = sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray()

    resultsfile = open(resdir + resifile(resi), 'w')
    resultsstr = ''
    resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
    resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
    resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
    resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
    resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
    resultsstr += 'xi['+str(resi)+']='+mathematica(xi)+';\n'
    resultsstr += 'Ws['+str(resi)+']='+mathematica(Ws)+';\n'
    resultsstr += 'ts['+str(resi)+']='+mathematica([JWi(Wi) for Wi in Ws])+';\n'
    resultsstr += 'Us['+str(resi)+']='+mathematica([UW(Wi) for Wi in Ws])+';\n'
    resultsstr += 'Ns['+str(resi)+']='+mathematica(Ns)+';\n'
    resultsstr += 'solved['+str(resi)+']='+mathematica(solved)+';\n'
    resultsstr += 'Eres['+str(resi)+']='+mathematica(Es)+';\n'
    resultsstr += 'nres['+str(resi)+']='+mathematica(ns)+';\n'
    resultsstr += 'n2res['+str(resi)+']='+mathematica(n2s)+';\n'
    resultsstr += 'corrres['+str(resi)+']='+mathematica(corrs)+';\n'
    resultsstr += 'ncorrres['+str(resi)+']='+mathematica(ncorrs)+';\n'
    resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
    resultsfile.write(resultsstr)

    # print >>sys.stderr, 'Res: ' + str(resi)

if __name__ == '__main__':
    main()