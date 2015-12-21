
from cmath import sqrt
import numpy as np
from parms import g, U0t, Ut, Jt
from numba import jit, int64, float64, complex128


@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S1b(f, i, j1, j2, ib, n, m, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + m == nb):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb))
	
	if (m == 1 + nb):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb)
	
	if (1 + n == nb):
		bf += f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,n]*f[j1,m]*g(n,m)*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb) - f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb)
	
	if (1 + n == nb):
		bf += f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j2,m]*g(n,m)*J[i]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb) - f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*J[i]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb)
	
	if (1 + m == nb):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*J[i]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb))
	
	if (m == 1 + nb):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*J[i]*pow(1 - m + n,-1.)*pow(U0,-1.)*sqrt(nb)
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S1S1b1(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and nb == 1 + p and 1 + m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + p and 1 + m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p and m == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + p and m == 1 + q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + m == nb and n == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == p and 1 + nb == q and m == 1 + q):
		bf += (f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and 1 + n == p and nb == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (n == 1 + p and 1 + m == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,nb]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p and 1 + m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + p and 1 + m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p and m == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + p and m == 1 + q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + m == nb and n == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,q]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == p and 1 + nb == q and m == 1 + q):
		bf += (f[i,1 + p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and 1 + n == p and nb == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (n == 1 + p and 1 + m == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,nb]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + m == nb and nb == 1 + p and 1 + n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,q]*f[j1,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + p and 1 + n == q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p and 1 + n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,q]*f[j1,nb]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p and 1 + n == q):
		bf += (f[i,-1 + q].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and m == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,q]*f[j1,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + n == q and 1 + nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and m == 1 + p and nb == 1 + q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + n == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))/2.
	
	if (1 + m == nb and nb == 1 + p and 1 + n == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j2,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + p and 1 + n == q):
		bf += -(f[i,-1 + q].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p and 1 + n == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j2,nb]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p and 1 + n == q):
		bf += (f[i,-1 + q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and m == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j2,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + n == q and 1 + nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and m == 1 + p and nb == 1 + q):
		bf += -(f[i,-1 + q].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + n == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,p]*g(n,m)*g(p,q)*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and 1 + nb == q):
		bf += (f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,q]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,p]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,nb]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and 1 + nb == q):
		bf += (f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,p]*f[j1,m]*f[j2,q]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == p and nb == 1 + q):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (n == 1 + p and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + q].conjugate()*f[i,p]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S1S1b2(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + m == nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,p]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,nb]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,p]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,q]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,p]*f[k1,nb]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,-1 + q].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + m == nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,p]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,m]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,nb]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,m]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,p]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and 1 + nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,q]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,p]*f[k2,nb]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + p and nb == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,-1 + q].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,q]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += (f[i,-1 + q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,q]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,q]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == q and 1 + nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[j2,p]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += (f[i,-1 + q].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + n == q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,q]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == q and 1 + nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == nb and nb == 1 + q):
		bf += -(f[i,-1 + q].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + n == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,p]*f[j2,m]*g(n,m)*g(p,q)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,q]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,q]*f[k1,nb]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and m == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and m == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + m == nb and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,q]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + nb == q and m == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + q].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + m == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[k1,p]*g(n,m)*g(p,q)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,q]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and 1 + m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,q]*f[k2,nb]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and m == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + q].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (nb == 1 + p and m == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + q].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + m == nb and 1 + nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,q]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + nb == q and m == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (m == 1 + nb and nb == 1 + q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + q].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	
	if (1 + m == q and nb == 1 + q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,nb]*f[k2,p]*g(n,m)*g(p,q)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(1 + p - q,-1.)*pow(U0,-2.)*sqrt(nb))/2.
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S21b1(f, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == nb and m == p and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == p and m == 2 + p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (m == p and n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == p and nb == p and m == 2 + p):
		bf += f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,p]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (m == p and n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + n == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == nb and m == p and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == p and m == 2 + p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (m == p and n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == p and nb == p and m == 2 + p):
		bf += f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,p]*f[j1,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (m == p and n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,p]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + n == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == nb and m == p and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == p and m == 2 + p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (m == p and n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == p and nb == p and m == 2 + p):
		bf += f[i,1 + p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,p]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (m == p and n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + n == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == nb and m == p and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == p and m == 2 + p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (m == p and n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p and m == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == p and nb == p and m == 2 + p):
		bf += f[i,1 + p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,p]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (m == p and n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (n == p and nb == p and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + m == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + m == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb))
	
	if (n == p and nb == p and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[j1],2.)*sqrt(nb)
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S21b2(f, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (n == p and nb == p and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + m == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + m == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (1 + n == nb and n == p and m == 1 + p):
		bf += -(f[i,p].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb))
	
	if (n == p and nb == p and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (n == p and m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,p]*g(n,m)*g(p,1 + p)*pow(1 - m + n,-2.)*pow(U0,-2.)*pow(J[i],2.)*sqrt(nb)
	
	if (1 + n == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and 1 + n == p):
		bf += f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + nb and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,p]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == p and nb == p):
		bf += f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (nb == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,p]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,p]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and 1 + n == p):
		bf += f[i,1 + p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,p]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + nb and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,p]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and 1 + n == p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,p]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == p and nb == p):
		bf += f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (nb == p and n == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,p]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == p and nb == 2 + p):
		bf += -(f[i,1 + p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,p]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S21b3(f, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and m == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,p]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,p]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,p]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,p]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (nb == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,p]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,p]*f[k1,nb]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,p]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == 1 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,p]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 1 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,p]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,p]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (nb == p and m == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,p]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,p]*f[k2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,1 + p].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == p):
		bf += f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p):
		bf += f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == p):
		bf += f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,n]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[j2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S21b4(f, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == p):
		bf += f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == p):
		bf += f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + m == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and n == p):
		bf += f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == 1 + p):
		bf += f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 1 + p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (n == p and nb == p):
		bf += f[i,-1 + nb].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (n == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,1 + p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,p]*f[j2,m]*g(n,m)*g(p,1 + p)*J[i]*J[j1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,nb]*f[j1,1 + p]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,1 + p].conjugate()*f[i,nb]*f[j1,m]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 2 + p):
		bf += f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,nb]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 2 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 2 + p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,m]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,1 + p].conjugate()*f[i,n]*f[j1,nb]*f[k1,p]*g(n,m)*g(p,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,nb]*f[j2,1 + p]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (1 + n == nb and m == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,1 + p].conjugate()*f[i,nb]*f[j2,m]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + n == nb and m == 2 + p):
		bf += f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,nb]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,nb]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 2 + p and nb == 1 + p):
		bf += f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == 2 + p and nb == 1 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,nb]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb))
	
	if (nb == p and m == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,m]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	
	if (m == p and nb == 2 + p):
		bf += f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,1 + p].conjugate()*f[i,n]*f[j2,nb]*f[k2,p]*g(n,m)*g(p,1 + p)*J[i]*J[j2]*pow(1 - m + n,-2.)*pow(U0,-2.)*sqrt(nb)
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S22b1(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and m == p and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 2 + p and 2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[i,nb]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[i,n]*f[j1,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[j1],2.)*sqrt(nb))/4.
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S22b2(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += (f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and m == q and nb == q):
		bf += -(f[i,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += -(f[i,p].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p and m == q):
		bf += (f[i,p].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == p and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*pow(J[i],2.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,-1 + q]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,1 + p]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,p].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,nb]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,nb]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (2 + n == q and nb == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and n == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and n == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,-1 + q]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and 2 + n == q):
		bf += -(f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and 2 + n == q):
		bf += (f[i,q].conjugate()*f[j1,-1 + m].conjugate()*f[j2,p].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S22b3(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,nb]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,nb]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,-1 + q]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[k1,p].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,nb]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += (f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == q):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,nb]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == q and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,1 + p]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,-1 + q]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == 2 + p and m == q):
		bf += (f[i,1 + n].conjugate()*f[j2,q].conjugate()*f[k2,p].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,nb]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,1 + p]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,n]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[j2,q].conjugate()*f[i,nb]*f[j1,m]*f[j2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	return bf

@jit(complex128(complex128[:,:],int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,int64,float64,float64,float64[:],float64[:]), nopython=True, cache=True)
def S22b4(f, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J):
	bf = 0
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + nb].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == q):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,nb]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,1 + p]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += -(f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and n == p):
		bf += (f[i,p].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,n]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (n == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,q].conjugate()*f[j2,-1 + m].conjugate()*f[i,nb]*f[j1,-1 + q]*f[j2,m]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,nb]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,nb]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,nb]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,nb]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += (f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,-1 + nb].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,nb]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,1 + p]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,p].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + nb].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,m]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j1,-1 + m].conjugate()*f[k1,q].conjugate()*f[i,n]*f[j1,nb]*f[k1,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[j1]*J[k1]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,nb]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,nb]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == p):
		bf += (f[i,-1 + nb].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,nb]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,nb]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += (f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + n == nb and m == 2 + p):
		bf += -(f[i,-1 + nb].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,-1 + nb].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 2 + p and nb == q):
		bf += (f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,nb]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == 1 + nb and m == p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,1 + p]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (1 + m == nb and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,p].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (nb == p and m == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + nb].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,m]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += (f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(1 - m + n,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	
	if (m == p and nb == 2 + p):
		bf += -(f[i,1 + n].conjugate()*f[j2,-1 + m].conjugate()*f[k2,q].conjugate()*f[i,n]*f[j2,nb]*f[k2,-1 + q]*g(n,m)*g(-1 + q,1 + p)*J[i]*J[j2]*pow(-1 - p + q,-1.)*pow(U0,-1.)*pow((1 - m + n)*U0 + (-1 - p + q)*U0,-1.)*sqrt(nb))/4.
	return bf

@jit(complex128[:](complex128[:],int64,int64,float64,float64,float64[:],float64), nopython=True, cache=True)
def b(f, L, nmax, mu, W, xi, scale):
	fin = f.reshape((L, nmax+1))
	U0 = U0t(W)
	Wi = W * xi
	U = Ut(Wi)
	dU = U - U0
	J = Jt(Wi)
	bf = np.zeros(L, dtype=np.complex128)


	for ib in range(L):
		for nb in range(nmax):
			for mb in range(1, nmax+1):
				if nb != mb-1:
					for i in range(L):
						k1 = (i-2) % L
						j1 = (i-1) % L
						j2 = (i+1) % L
						k2 = (i+2) % L
						for n in range(nmax):
							for m in range(1, nmax+1):
								if n != m-1:
									bf[ib] += S1b(fin, i, j1, j2, ib, n, m, nb, mb, mu, U0, dU, J)
									for p in range(nmax):
										bf[ib] += S21b1(fin, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J)
										bf[ib] += S21b2(fin, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J)
										bf[ib] += S21b3(fin, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J)
										bf[ib] += S21b4(fin, i, j1, j2, k1, k2, ib, n, m, p, nb, mb, mu, U0, dU, J)
										for q in range(1, nmax+1):
											if p != q-1:
												bf[ib] += S1S1b1(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)
												bf[ib] += S1S1b2(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)
												if n-m != p-q:
													bf[ib] += S22b1(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)
													bf[ib] += S22b2(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)
													bf[ib] += S22b3(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)
													bf[ib] += S22b4(fin, i, j1, j2, k1, k2, ib, n, m, p, q, nb, mb, mu, U0, dU, J)

	return bf
