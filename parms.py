from __future__ import division
from math import sqrt, pow
from numba import jit, int64, float64
import numpy as np

@jit(float64(int64, int64), nopython=True, cache=False)
def g(n ,m):
    return sqrt(m*(n+1))

@jit(float64(float64,float64,float64,float64), nopython=True, cache=False)
def Wt(t, Wi, Wf, tau):
    if t < tau:
        return Wi + (Wf - Wi) * t / tau
    else:
        return Wf + (Wi - Wf) * (t - tau) / tau

@jit(float64(float64,float64,float64,float64), nopython=True, cache=False)
def Wtp(t, Wi, Wf, tau):
    if t < tau:
        return (Wf - Wi) / tau
    else:
        return (Wi - Wf) / tau

@jit(float64(float64), nopython=True, cache=False)
def Ui(W):
    return (3.9062500000000004e30*W**2)/((6.250000000000001e21 + W**2)**2)

@jit(float64(float64,float64), nopython=True, cache=False)
def Jij(Wi, Wj):
    return (1.1e7*Wi*Wj)/(sqrt(6.250000000000001e21 + Wi**2)*sqrt(6.250000000000001e21 + Wj**2))

@jit(float64(float64), nopython=True, cache=False)
def U0t(W):
    return Ui(W)

@jit(float64[:](float64[:]), nopython=True, cache=False)
def Ut(W):
    L = len(W)
    U = np.zeros(L)
    for i in range(L):
        U[i] = Ui(W[i])
    return U

@jit(float64[:](float64[:]), nopython=True, cache=False)
def Jt(W):
    L = len(W)
    J = np.zeros(L)
    for i in range(L):
        J[i] = Jij(W[i], W[(i+1)%L])
    return J

@jit(float64(float64,float64), nopython=True, cache=False)
def U0tp(W, Wp):
    return (-1.5625000000000001e31*W**3*Wp)/((6.250000000000001e21 + W**2)**3) + (7.812500000000001e30*W*Wp)/((6.250000000000001e21 + W**2)**2)

@jit(float64(float64,float64,float64,float64), nopython=True, cache=False)
def Jpij(Wi, Wj, Wip, Wjp):
    return (-1.1e7*Wi**2*Wj*Wip)/((6.250000000000001e21 + Wi**2)**(3/2)*sqrt(6.250000000000001e21 + Wj**2)) + (1.1e7*Wj*Wip)/(sqrt(6.250000000000001e21 + Wi**2)*sqrt(6.250000000000001e21 + Wj**2)) -\
    (1.1e7*Wi*Wj**2*Wjp)/(sqrt(6.250000000000001e21 + Wi**2)*(6.250000000000001e21 + Wj**2)**(3/2)) + (1.1e7*Wi*Wjp)/(sqrt(6.250000000000001e21 + Wi**2)*sqrt(6.250000000000001e21 + Wj**2))

@jit(float64[:](float64[:],float64[:]), nopython=True, cache=False)
def Jtp(W, Wp):
    L = len(W)
    J = np.zeros(L)
    for i in range(L):
        J[i] = Jpij(W[i], W[(i+1)%L], Wp[i], Wp[(i+1)%L])
    return J

