from E import E
from odes import odes
from b import b
from parms import Ui
import numpy as np
from scipy.integrate import ode
from timeit import timeit
import concurrent.futures
import sys
import os
import datetime
import struct
import progressbar

def mathematica(x):
    if x is 'True' or x is 'False':
        return x
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        try:
            return '{:.16}'.format(x).replace('j', 'I').replace('e', '*^')
        except:
            return str(x)

def resifile(i):
    return 'res.' + str(i) + '.txt'

def odesfwd(t, f, L, nmax, mu, Wi, Wf, tau, xi):
    return odes(t, f, L, nmax, mu, Wi, Wf, tau, xi)

L = 5
nmax = 7
dim = nmax+1

Wi = 3e11
Wf = 1e11
mu = 0.5*Ui(Wi)
np.random.seed()
xi = np.zeros(L)

def runtau(f0, E0, tau):
    r = ode(odesfwd).set_integrator('zvode', method='bdf', nsteps=1000000)
    r.set_initial_value(f0).set_f_params(L, nmax, mu, Wi, Wf, tau, xi)
    tf = 2*tau
    r.integrate(tf)
    ff = r.y
    Ef = E(ff, L, nmax, mu, Wi, xi)
    Q = Ef - E0
    f0m = f0.reshape((L, nmax+1))
    ffm = ff.reshape((L, nmax+1))
    p = sum([1 - abs(np.vdot(ffi, f0i))**2 for (ffi, f0i) in zip(ffm, f0m)])/L
    b0 = b(f0, L, nmax, mu, Wi, xi)
    bf = b(ff, L, nmax, mu, Wi, xi)
    return [r.successful(), tau, ff, Ef, Q, p, b0, bf]

nthreads = 35

resi =int(sys.argv[2])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/CTDG Numba/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

seed = int(sys.argv[1])
Delta = 0
def main():
    start = datetime.datetime.now()

    f0 = np.zeros(L*dim, dtype=complex)

    inputdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/CTDG Input/')
    with open(inputdir + "input_" + str(L) + "_" + str(seed) + "_" + str(Delta) + ".bin") as f:
        for i in range(L):
            double = f.read(8)
            xi[i] = struct.unpack('d', double)[0]
        for i in range(L*dim):
            re = f.read(8)
            im = f.read(8)
            f0[i] = struct.unpack('d', re)[0] + struct.unpack('d', im)[0]*1j

    E0 = E(f0, L, nmax, mu, Wi, xi)

    taus = np.linspace(1e-7, 3.23529e-7, nthreads)
    ntaus = len(taus)

    pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.Timer()], maxval=ntaus).start()

    success = []
    taures = []
    ffres = []
    Efres = []
    Qres = []
    pres = []
    b0res = []
    bfres = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
        futures = [executor.submit(runtau, f0, E0, tau) for tau in taus]
        for future in pbar(concurrent.futures.as_completed(futures)):
            result = future.result()
            success.append(result[0])
            taures.append(result[1])
            ffres.append(result[2])
            Efres.append(result[3])
            Qres.append(result[4])
            pres.append(result[5])
            b0res.append(result[6])
            bfres.append(result[7])

    success = [x for (y, x) in sorted(zip(taures, success))]
    ffres = [x for (y, x) in sorted(zip(taures, ffres))]
    Efres = [x for (y, x) in sorted(zip(taures, Efres))]
    Qres = [x for (y, x) in sorted(zip(taures, Qres))]
    pres = [x for (y, x) in sorted(zip(taures, pres))]
    b0res = [x for (y, x) in sorted(zip(taures, b0res))]
    bfres = [x for (y, x) in sorted(zip(taures, bfres))]
    taures.sort()

    end = datetime.datetime.now()

    resultsfile = open(resdir + 'res.'+str(resi)+'.txt', 'w')
    resultsstr = ''
    resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
    resultsstr += 'xires['+str(resi)+']='+mathematica(xi)+';\n'
    resultsstr += 'f0res['+str(resi)+']='+mathematica(f0)+';\n'
    resultsstr += 'E0res['+str(resi)+']='+mathematica(E0)+';\n'
    resultsstr += 'success['+str(resi)+']='+mathematica(success)+';\n'
    resultsstr += 'taures['+str(resi)+']='+mathematica(taures)+';\n'
    resultsstr += 'ffres['+str(resi)+']='+mathematica(ffres)+';\n'
    resultsstr += 'Efres['+str(resi)+']='+mathematica(Efres)+';\n'
    resultsstr += 'Qres['+str(resi)+']='+mathematica(Qres)+';\n'
    resultsstr += 'pres['+str(resi)+']='+mathematica(pres)+';\n'
    resultsstr += 'b0res['+str(resi)+']='+mathematica(b0res)+';\n'
    resultsstr += 'bfres['+str(resi)+']='+mathematica(bfres)+';\n'
    resultsstr += 'runtime['+str(resi)+']="'+str(end-start)+'";\n'
    resultsfile.write(resultsstr)

    print 'Res: ' + str(resi)

if __name__ == '__main__':
    main()
