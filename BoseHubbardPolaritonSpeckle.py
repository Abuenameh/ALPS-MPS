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
import shutil
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

speckles = {}

seed = int(sys.argv[1])

def speckle(W, sigma):
    if speckles.has_key(W):
        return speckles[W]

    if sigma == 0:
        speckleW = np.zeros(L)
        speckleW.fill(W)
        speckles[W] = speckleW
        return speckleW

    np.random.seed(seed)

    FFTD = 200
    FFTL = int(sigma * FFTD)

    A = (4 / np.pi) * (W / FFTD)
    a = [[A * np.exp(2 * np.pi * np.random.random() * 1j) if (i * i + j * j < 0.25 * FFTD * FFTD) else 0 for i in
          range(-FFTL // 2, FFTL // 2, 1)] for j in range(-FFTL // 2, FFTL // 2, 1)]

    b = np.fft.fft2(a)
    s = np.real(b * np.conj(b))
    s2 = np.sqrt(s)
    speckleW = s2.flatten()[0:L]
    speckles[W] = speckleW
    return speckleW

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

numthreads = 4

L = 25
nmax = 5
sweeps = 50
maxstates = 200

#prepare the input parameters
parms = OrderedDict()
# parms['SEED'] = np.random.rand()
parms['LATTICE_LIBRARY'] = 'lattice' + str(L) + '.xml'
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
parms['MEASURE[Entropy]'] = 1
parms['MEASURE[Renyi2]'] = 1
parms['entanglement_spectra'] = ','.join([str(i) for i in np.arange(0,L)])
# parms['init_state'] = 'local_quantumnumbers'
parms['chkp_each'] = sweeps
parms['N_total'] = L

bounds = tuple([(0, nmax)] * L)

# seed = int(sys.argv[1])
# np.random.seed(seed)
# ximax = float(sys.argv[2])
# xi = (1 + ximax * np.random.uniform(-1, 1, L))
# # xi = np.array([1.0488135039273247529, 1.2151893663724195882, 1.1027633760716439859, \
# # 1.04488318299689675328, 0.9236547993389047084])
# # xi = np.array([1]*L)
# xisort = sorted([(xii, i) for (i, xii) in enumerate(xi)])

resi = int(sys.argv[3])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

try:
    shutil.rmtree('Tasks')
    os.mkdir('Tasks')
except:
    pass

basename = 'Tasks/bhp.' + str(L) + '.' + str(resi) + '.'

def runmps(task, iW, isigma, Wi, sigma):
    parmsi = deepcopy(parms)
    parmsi['iW'] = iW
    parmsi['is'] = isigma

    t = JW(speckle(Wi, sigma))
    U = UW(speckle(Wi, sigma))
    for i in range(L):
        parmsi['t'+str(i)] = t[i]
    for i in range(L):
        parmsi['U'+str(i)] = U[i]

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
    # except:
    #     basen = N // L
    #     ns = [basen] * L
    #     rem = N % L
    #     excessi = [i for (xii, i) in xisort[:rem]]
    #     for i in excessi:
    #         ns[i] += 1
    # parmsi['initial_local_N'] = ','.join([str(n) for n in ns])

    input_file = pyalps.writeInputFiles(basename + str(task), [parmsi])
    pyalps.runApplication('mps_optim', input_file, writexml=True)

def main():
    Ws = np.linspace(7.9e10, 1.1e12, 10)#[1e11]#[7.9e10]#np.linspace(2e11,3.2e11,10)#[2e10]
    nW = len(Ws)
    # Ns = [L]#range(0,2*L+1)#range(30,86)#[L]#range(0,2*L+1)#range(40,70)#range(0,2*L+1)#range(24,2*L+1)#range(0,2*L+1)#range(23,27)
    # nN = len(Ns)
    sigmas = [1,2]#range(0, 11)
    nsigma = len(sigmas)
    Wsigmas = zip(range(nW*nsigma), [[i, j] for i in range(nW) for j in range(nsigma)], [[Wi, sigmai] for Wi in Ws for sigmai in sigmas])
    ntasks = len(Wsigmas)

    start = datetime.datetime.now()

    pbar = progressbar.ProgressBar(widgets=['Res: '+str(resi)+' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.Timer()], maxval=ntasks).start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(runmps, task, iW, isigma, Wi, sigma) for (task, [iW, isigma], [Wi, sigma]) in Wsigmas]
        for future in pbar(concurrent.futures.as_completed(futures)):
            future.result()
            sys.stderr.flush()

    end = datetime.datetime.now()

    #load all measurements for all states
    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

    Es = makeres(nW, nsigma)
    ns = makeres(nW, nsigma)
    n2s = makeres(nW, nsigma)
    corrs = makeres(nW, nsigma)
    ncorrs = makeres(nW, nsigma)
    entropy = makeres(nW, nsigma)
    es = makeres(nW, nsigma)
    for d in data:
        for sigma in d:
            iW = int(sigma.props['iW'])
            isigma = int(sigma.props['is'])
            if(sigma.props['observable'] == 'Energy'):
                Es[iW][isigma] = sigma.y[0]
            if(sigma.props['observable'] == 'Local density'):
                ns[iW][isigma] = sigma.y[0]
            if(sigma.props['observable'] == 'Local density squared'):
                n2s[iW][isigma] = sigma.y[0]
            if(sigma.props['observable'] == 'One body density matrix'):
                corrs[iW][isigma] = sparse.coo_matrix((sigma.y[0], (sigma.x[:,0], sigma.x[:,1]))).toarray()
            if(sigma.props['observable'] == 'Density density'):
                ncorrs[iW][isigma] = sparse.coo_matrix((sigma.y[0], (sigma.x[:,0], sigma.x[:,1]))).toarray()
            if(sigma.props['observable'] == 'Entropy'):
                entropy[iW][isigma] = sigma.y[0]
            if(sigma.props['observable'] == 'Entanglement Spectra'):
                es[iW][isigma] = [[sigma for sigma in reversed(sorted(esi[1]))][0:4] for esi in sigma.y[0]]

    resultsfile = open(resdir + resifile(resi), 'w')
    resultsstr = ''
    resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
    resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
    resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
    resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
    resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
    resultsstr += 'Ws['+str(resi)+']='+mathematica(Ws)+';\n'
    resultsstr += 'ts['+str(resi)+']='+mathematica([JWi(Wi) for Wi in Ws])+';\n'
    resultsstr += 'Us['+str(resi)+']='+mathematica([UW(Wi) for Wi in Ws])+';\n'
    resultsstr += 'sigmas['+str(resi)+']='+mathematica(sigmas)+';\n'
    resultsstr += 'Eres['+str(resi)+']='+mathematica(Es)+';\n'
    resultsstr += 'nres['+str(resi)+']='+mathematica(ns)+';\n'
    resultsstr += 'n2res['+str(resi)+']='+mathematica(n2s)+';\n'
    resultsstr += 'corrres['+str(resi)+']='+mathematica(corrs)+';\n'
    resultsstr += 'ncorrres['+str(resi)+']='+mathematica(ncorrs)+';\n'
    resultsstr += 'entropy['+str(resi)+']='+mathematica(entropy)+';\n'
    resultsstr += 'es['+str(resi)+']='+mathematica(es)+';\n'
    resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
    resultsfile.write(resultsstr)

    # print >>sys.stderr, 'Res: ' + str(resi)

if __name__ == '__main__':
    main()