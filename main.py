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
    return 'res.' + str(i) + '.txt'

def odesfwd(t, f, L, nmax, mu, Wi, Wf, tau, xi, scale):
    return odes(t, f, L, nmax, mu, Wi, Wf, tau, xi, scale)

L = 5
nmax = 7
dim = nmax+1

Wi = 3e11
Wf = 1e11
mu = 0.5*Ui(Wi)
np.random.seed()
xi = np.zeros(L)
scale = 1

def runtau(f0, E0, tau):
    r = ode(odesfwd).set_integrator('zvode', method='bdf', nsteps=10000)
    r.set_initial_value(f0).set_f_params(L, nmax, mu, Wi, Wf, tau, xi, scale)
    tf = 2*tau
    r.integrate(tf)
    ff = r.y
    Ef = E(ff, L, nmax, mu, Wi, xi, scale)
    Q = Ef - E0
    f0m = f0.reshape((L, nmax+1))
    ffm = ff.reshape((L, nmax+1))
    p = sum([1 - abs(np.vdot(ffi, f0i))**2 for (ffi, f0i) in zip(ffm, f0m)])/L
    b0 = b(f0, L, nmax, mu, Wi, xi, scale)
    bf = b(ff, L, nmax, mu, Wi, xi, scale)
    return [r.successful(), tau, ff, Ef, Q, p, b0, bf]

nthreads = 2

resi = 0#int(sys.argv[3])
resdir = os.path.expanduser('~/Dropbox/Amazon EC2/Simulation Results/CTDG Numba/')
while os.path.exists(resdir + resifile(resi)):
    resi += 1

seed = int(sys.argv[1])
Delta = 0.1
def main():
    start = datetime.datetime.now()

    f0 = np.zeros(L*dim, dtype=complex)

    inputdir = os.path.expanduser('~/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2/')
    with open(inputdir + "input_" + str(L) + "_" + str(seed) + "_" + str(Delta) + ".bin") as f:
        for i in range(L):
            double = f.read(8)
            xi[i] = struct.unpack('d', double)[0]
        for i in range(L*dim):
            re = f.read(8)
            im = f.read(8)
            f0[i] = struct.unpack('d', re)[0] + struct.unpack('d', im)[0]*1j

    E0 = E(f0, L, nmax, mu, Wi, xi, scale)

    taus = np.linspace(1e-11, 1.1e-11, nthreads)
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(runtau, f0, E0, scale*tau) for tau in taus]
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

    end = datetime.datetime.now()

    resultsfile = open(resdir + 'res.'+str(resi)+'.txt', 'w')
    resultsstr = ''
    resultsstr += 'seed['+str(resi)+']='+str(seed)+';\n'
    resultsstr += 'xires['+str(resi)+']='+mathematica(xi)+';\n'
    resultsstr += 'scale['+str(resi)+']='+mathematica(scale)+';\n'
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
