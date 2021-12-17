# GR4J hydrological bucket model after Fabrizo Fenicia

import numpy as np
from numba import jit

# HELPER FUNCTIONS:
def SS1(i,tBase):
    if (i<=0):
        y=0
    elif (i<tBase):
        y=(i/tBase)**2.5
    else:
        y=1
    return y

def SS2(i,tBase):
    if (i<=0):
        y=0
    elif (i<tBase):
        y=0.5*(i/tBase)**2.5
    elif (i < 2*tBase):
        y = 1-0.5 * (2-i / tBase) ** 2.5
    else:
        y=1
    return y

def tWeightsL1(tBase):
    iMax = int(np.ceil(tBase))
    w = np.zeros((iMax,1))
    for i in np.arange(iMax):
        w[i]=SS1(i,tBase)-SS1(i-1,tBase)
    return w

def tWeightsL2(tBase):
    iMax = int(np.ceil(tBase*2))
    w = np.zeros((iMax,1))
    for i in np.arange(iMax):
        w[i]=SS2(i,tBase)-SS2(i-1,tBase);
    return w

# STORAGE FUNCTIONS
def fun_Qq_UR(S,Pn,x1):
    return (x1*(1-(S/x1)**2)*np.tanh(Pn/x1))/(1+S/x1*np.tanh(Pn/x1))

def fun_Qp_UR(S,x1):
    return S*(1-(1+(4/9.*S/x1)**4)**-0.25)

def fun_E_UR(S,Ep,x1):
    return (S*(2-S/x1)*np.tanh(Ep/x1))/(1+(1-S/x1)*np.tanh(Ep/x1))



def pyGR4J(data,para,w1,w2,start_i=-99,nT=-99):
    # forcings
    P = data[:,0]
    Epot = data[:,1]

    par_SiniFr_UR = 0.
    par_x1,par_x2,par_x3,par_x4,par_spl = para

    if start_i == -99:
        start_i = 0
    if nT == -99:
        nT = len(data) - start_i

    #allocate reservoirs
    st_UR = np.zeros(nT)*np.nan   #production store
    st_RR = np.zeros(nT) * np.nan #routing store

    lW1 = len(w1)
    lW2 = len(w2)
    stLag1 = np.zeros(lW1) # store lag function values
    stLag2 = np.zeros(lW2) # store lag function values

    #fluxes
    Eact = np.zeros(nT)*np.nan
    Qsim = np.zeros(nT)*np.nan

    #initialize states
    stStart_UR = par_SiniFr_UR * par_x1
    stStart_RR = 0.

    st_UR[0] = stStart_UR
    st_RR[0] = stStart_RR

    ###
    # RUN MODEL
    for i in np.arange(nT):
        # interception
        E_IR = np.min([Epot[i], P[i]])
        E_URpot = Epot[i] - E_IR
        P_UR = P[i] - E_IR
        # production store
        Qq_UR = P_UR - fun_Qq_UR(st_UR[i], P_UR, par_x1)
        E_UR = fun_E_UR(st_UR[i], E_URpot, par_x1)
        st_UR[i] = st_UR[i] + P_UR - Qq_UR - E_UR
        Eact[i] = E_IR + E_UR
        # percolation
        Qp_UR = fun_Qp_UR(st_UR[i], par_x1)
        st_UR[i] = st_UR[i] - Qp_UR
        # split
        P_SP = Qp_UR + Qq_UR
        P_L1 = par_spl * P_SP
        P_L2 = P_SP - P_L1
        # lag 1
        for k in np.arange(lW1 - 1):
            stLag1[k] = stLag1[k + 1] + w1[k] * P_L1
        stLag1[lW1 - 1] = w1[lW1 - 1] * P_L1
        # lag 2
        for k in np.arange(lW2 - 1):
            stLag2[k] = stLag2[k + 1] + w2[k] * P_L2
        stLag2[lW2 - 1] = w2[lW2 - 1] * P_L2
        # Water exchange
        Q_EX = par_x2 * (st_RR[i] / par_x3) ** 3.5
        # Routing store
        st_RR[i] = np.max([0., st_RR[i] + stLag1[0] + Q_EX])
        R2 = st_RR[i] / (1 + (st_RR[i] / par_x3) ** 4) ** 0.25
        Q_RR = st_RR[i] - R2
        st_RR[i] = R2
        # Output
        QD = np.max([0., stLag2[0] + Q_EX])
        Qsim[i] = Q_RR + QD
        if (i < nT - 1):
            st_UR[i + 1] = st_UR[i]
            st_RR[i + 1] = st_RR[i]

    return Qsim