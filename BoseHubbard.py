__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time
from copy import deepcopy

basename = 'Tasks/bh'+str(time.time())

L = 50

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice.xml'
parms['LATTICE'] = 'inhomogeneous open chain lattice'
# parms['LATTICE'] = 'open chain lattice'
parms['MODEL_LIBRARY'] = 'model.xml'
parms['MODEL'] = 'boson Hubbard'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = 5
parms['SWEEPS'] = 100
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = 200
parms['MEASURE_LOCAL[Local density]'] = 'n'
parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'

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

basename = 'Tasks/bh2'

parmslist = []
for N in range(0, L+1):
    parmsi = deepcopy(parms)
    parmsi['N_total'] = N
    parmslist.append(parmsi)


#write the input file and run the simulation
input_file = pyalps.writeInputFiles(basename,parmslist)
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

energies = []
for i in range(0,len(data)):
    for s in data[i]:
        if(s.props['observable'] == 'Energy'):
            energies.append(s.y[0])

energyfile = open('/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/ALPS-MPS/energies.txt', 'w')
energiesstr = '{' + ','.join(["{:.20f}".format(en) for en in energies]) + '}'
energyfile.write(energiesstr)
