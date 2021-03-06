__author__ = 'ubuntu'

import pyalps
import pyalps.plot
import numpy as np
from collections import OrderedDict
import os
import shutil
import matplotlib.pyplot as plt

L = 10
nmax = 5
sweeps = 1000
maxstates = 200

#prepare the input parameters
parms = OrderedDict()
parms['LATTICE'] = 'open chain lattice'
# parms['LATTICE'] = 'inhomogeneous chain lattice'
parms['MODEL'] = 'boson Hubbard'
# parms['MODEL'] = 'fermion Hubbard'
# parms['MODEL'] = 'hardcore boson'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
# parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = nmax
parms['SWEEPS'] = sweeps
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = maxstates
parms['MEASURE_LOCAL[Local density]'] = 'n'
# parms['MEASURE_LOCAL[Local density squared]'] = 'n2'
# parms['MEASURE_CORRELATIONS[One body density matrix]'] = 'bdag:b'
# parms['MEASURE_CORRELATIONS[Density density]'] = 'n:n'
# parms['init_state'] = 'local_quantumnumbers'
parms['chkp_each'] = sweeps
parms['optimization'] = 'singlesite'
# parms['optimization'] = 'twosite'
parms['ngrowsweeps'] = 200
parms['nmainsweeps'] = 200
# parms['alpha_initial'] = 1e-4
# parms['alpha_main'] = 1e-6
# parms['alpha_final'] = 1e-8
# parms['storagedir'] = 'SingleSite/storage'

parms['t'] = 0.3
parms['U'] = 1

resi = 1000
try:
    shutil.rmtree('SingleSite')
    os.mkdir('SingleSite')
except:
    pass

basename = 'SingleSite/ss.'

parms['N_total'] = 14
# parms['initial_local_N'] = '3,2,2,2,2,2,2,2,2,2'#'1,1,1,1,1,1,1,1,0,0'#'2,2,1,1,1,1,1,1,1,1'
input_file = pyalps.writeInputFiles(basename + str(resi), [parms])
pyalps.runApplication('mps_optim', input_file, writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

for d in data:
    for s in d:
        if(s.props['observable'] == 'Energy'):
            print s.y[0]
        if(s.props['observable'] == 'Local density'):
            print s.y[0]

iters = pyalps.loadIterationMeasurements(pyalps.getResultFiles(prefix=basename), what=['Energy'])
# print iters
en_vs_iter = pyalps.collectXY(iters, x='iteration', y='Energy')
# print en_vs_iter
Es = np.array([Ei for (i, Ei) in sorted(zip([int(xi) for xi in en_vs_iter[0].x[0:-1:20]], en_vs_iter[0].y[0:-1:20]))[-40:-1]])
Es2 = Es[1:-1] - Es[0:-2]
print Es2
plt.figure()
# pyalps.plot.plot(en_vs_iter)
plt.plot(Es)
# plt.yscale('log')
plt.show()