__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time
from copy import deepcopy
from scipy import sparse
import datetime

def mathematica(x):
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        return '{:.20f}'.format(x)

basename = 'Tasks/bh'+str(time.time())

L = 50
nmax = 5
sweeps = 100
maxstates = 200

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice50.xml'
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
parms['init_state'] = 'local_quantumnumbers'
# parms['initial_local_N'] = '1,1,1,2,1,1,1,2,2,1,1,1'#'2,2,2,1,1,1,1,1,1,1,1,1'

seed = 0
np.random.seed(seed)
t = (1 + 0.5 * np.random.uniform(-1, 1, L-1)) * 0.01 #[0.01, 0.02, 0.03, 0.02]
U = (1 + 0.5 * np.random.uniform(-1, 1, L)) * 1 #[1, 2, 3, 1, 2]
# parms['ts'] = mathematica(t)#["{:.20f}".format(ti) for ti in t]
# parms['Us'] = mathematica(U)#["{:.20f}".format(Ui) for Ui in U]
for i in range(L-1):
    parms['t'+str(i)] = t[i]
for i in range(L):
    parms['U'+str(i)] = U[i]

Usort = sorted([(Ui, i) for (i, Ui) in enumerate(U)])

basename = 'Tasks/bh.50.'+str(seed)
resi = 2
basename = 'Tasks/bh.50.'+str(resi)

parmslist = []
# for N in range(L+1, 2*L+1):
# for N in [L+3]:
for N in range(0, 2*L+1):
    parmsi = deepcopy(parms)
    parmsi['N_total'] = N
    basen = N // L
    localqn = [basen] * L
    rem = N % L
    excessi = [i for (Ui, i) in Usort[:rem]]
    for i in excessi:
        localqn[i] += 1
    parmsi['initial_local_N'] = ','.join([str(n) for n in localqn])
    # parmsi['initfile'] = '/home/ubuntu/PycharmProjects/ALPS-MPS/Tasks/bh.12.10.task'+str(N-L)+'.out.chkp'
    parmslist.append(parmsi)


start = datetime.datetime.now()

#write the input file and run the simulation
input_file = pyalps.writeInputFiles(basename,parmslist)
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

end = datetime.datetime.now()

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))


Eresults = []
nresults = []
n2results = []
corrresults = []
ncorrresults = []
for d in data:
    for s in d:
        if(s.props['observable'] == 'Energy'):
            Eresults += [(s.props['N_total'], s.y[0])]
        if(s.props['observable'] == 'Local density'):
            nresults += [(s.props['N_total'], s.y[0])]
        if(s.props['observable'] == 'Local density squared'):
            n2results += [(s.props['N_total'], s.y[0])]
        if(s.props['observable'] == 'One body density matrix'):
            corrresults += [(s.props['N_total'], sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray())]
        if(s.props['observable'] == 'Density density'):
            ncorrresults += [(s.props['N_total'], sparse.coo_matrix((s.y[0], (s.x[:,0], s.x[:,1]))).toarray())]


Ns = [res[0] for res in sorted(Eresults)]
Es = [res[1] for res in sorted(Eresults)]
ns = [res[1] for res in sorted(nresults)]
n2s = [res[1] for res in sorted(n2results)]
corrs = [res[1] for res in sorted(corrresults)]
ncorrs = [res[1] for res in sorted(ncorrresults)]

# resultsfile = open('/Users/Abuenameh/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/res.'+str(resi)+'.txt', 'w')
resultsfile = open('/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/res.'+str(resi)+'.txt', 'w')
resultsstr = ''
resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
resultsstr += 'L['+str(resi)+']='+str(L)+';\n'
resultsstr += 'nmax['+str(resi)+']='+str(nmax)+';\n'
resultsstr += 'sweeps['+str(resi)+']='+str(sweeps)+';\n'
resultsstr += 'maxstates['+str(resi)+']='+str(maxstates)+';\n'
resultsstr += 'ts['+str(resi)+']='+mathematica(t)+';\n'
resultsstr += 'Us['+str(resi)+']='+mathematica(U)+';\n'
resultsstr += 'Ns['+str(resi)+']='+mathematica(Ns)+';\n'
resultsstr += 'Eres['+str(resi)+']='+mathematica(Es)+';\n'
resultsstr += 'nres['+str(resi)+']='+mathematica(ns)+';\n'
resultsstr += 'n2res['+str(resi)+']='+mathematica(n2s)+';\n'
resultsstr += 'corrres['+str(resi)+']='+mathematica(corrs)+';\n'
resultsstr += 'ncorrres['+str(resi)+']='+mathematica(ncorrs)+';\n'
resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
resultsfile.write(resultsstr)
# print(resultsstr)
