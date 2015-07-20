__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time
from copy import deepcopy

basename = 'Tasks/bh'+str(time.time())

L = 15

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice15.xml'
parms['LATTICE'] = 'inhomogeneous open chain lattice'
# parms['LATTICE'] = 'open chain lattice'
parms['MODEL_LIBRARY'] = 'model.xml'
parms['MODEL'] = 'boson Hubbard'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = 7
parms['SWEEPS'] = 100
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = 200
parms['MEASURE_LOCAL[Local density]'] = 'n'
parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'
# parms['optimization'] = 'singlesite'

np.random.seed(0)
t = (1 + 0.5 * np.random.uniform(-1, 1, L-1)) * 0.01 #[0.01, 0.02, 0.03, 0.02]
U = (1 + 0.5 * np.random.uniform(-1, 1, L)) * 1 #[1, 2, 3, 1, 2]
parms['ts'] = ["{:.20f}".format(ti) for ti in t]
parms['Us'] = ["{:.20f}".format(Ui) for Ui in U]
for i in range(L-1):
    parms['t'+str(i)] = t[i]
for i in range(L):
    parms['U'+str(i)] = U[i]

parms['N_total'] = 1

basename = 'Tasks/bhstestts3'
# basename = 'Tasks/bhq1'

parmslist = []
for N in range(L+1, 2*L+1):
    parmsi = deepcopy(parms)
    parmsi['N_total'] = N
    parmslist.append(parmsi)


#write the input file and run the simulation
input_file = pyalps.writeInputFiles(basename,parmslist)
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

results = []
for d in data:
    for s in d:
        if(s.props['observable'] == 'Energy'):
            results += [(s.props['N_total'], s.y[0])]

Ns = [res[0] for res in sorted(results)]
energies = [res[1] for res in sorted(results)]
# print(energies)

resultsfile = open('/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/Results/'+basename.split('/')[-1]+'.txt', 'w')
resultsstr = '{'+str(L)+',{'+','.join(["{:d}".format(int(N)) for N in Ns]) + '},{' + ','.join(["{:.20f}".format(en) for en in energies]) + '}}'
print(resultsstr)
resultsfile.write(resultsstr)

# energyfile = open('/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/energies2.txt', 'w')
# energiesstr = '{' + ','.join(["{:.20f}".format(en) for en in energies]) + '}'
# energyfile.write(energiesstr)
