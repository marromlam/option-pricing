import numpy as np

from GeomBrownian import geometric_brownian

# TODO: create generic function to generate determinitic seq of 
# random numbers with Box-Muller

def payoff_basquet_call(S=None, *args, **kwargs):
    V = max(max(args[0] * S[0] - kwargs['K'],
                args[1] * S[1] - kwargs['K']), 0)
    return V


def monte_carlo_pricing(nT, nPath, r, T, S0, delta, R, D, K, payoff, iPayOff,
               Par, Sigma, iVC=None, *args, **kwargs): 
    # TODO: remove this since it is just for developing purposes
    np.random.seed(0)
    S, Sa = geometric_brownian(S0, nPath, nT, T, Sigma, r, delta, D,
                               antithetic=True)
    if 1 == iPayOff:
        V = vanilla(S,Sa,nT, nPath, r, T, S0, delta, R, D, K, payoff, iPayOff,
                    Par, Sigma, iVC)
    elif 2 == iPayOff:
        V = holding_period_return(S,Sa,nT, nPath, r, T, S0, delta, R, D, K,
                                payoff, iPayOff, Par, Sigma, iVC)
    elif 3 == iPayOff:
        V = barrier(S,nT, nPath, r, T, S0, delta, R, D, K, payoff, iPayOff,
                    Par, Sigma, iPayoff)
    else:
        print('Not available option')
        exit()
    return V


def holding_period_return(S, Sa, nT, nPath, r, T, S0, delta, R, D, K, payoff,
                          iPayOff, Par, Sigma,ic = None): 
    #  The payoff only depends on S/S(0) values
    Pay = np.zeros((nPath,1))
    for j in np.arange(0,nPath).reshape(-1):
        Pay[j] = payoff(S[j,:,:], *Par)
        if ic == 2:
            Pay[j] = (Pay[j] + payoff(Sa[j,:,:],*Par)) / 2
    V = np.exp(- r * T) * mean(Pay)
    return V


def vanilla(S,Sa,nT, nPath, r, T, S0, delta, R, D, K, payoff, iPayOff,
            Par, Sigma,ic = None): 
    #  The payoff only depends on S values at t=T
    Pay = np.zeros((nPath,1))
    for j in np.arange(0,nPath).reshape(-1):
        Pay[j] = payoff(S[j,:,-1], *Par, K=K)
        if ic == 2:
            Pay[j] = (Pay(j) + payoff(Sa[j,:,-1], *Par, K=K)) / 2
    V = np.exp(- r * T) * np.mean(Pay)
    return V


# def Barrier(S = None,Opt = None): 
#     Pay = np.zeros((nPath,1))
#     
#     nHit = 0
#     
#     #-------------------------------------------------------------------------
#     for j in np.arange(1,nPath+1).reshape(-1):
#         Hitting = Hit(np.squeeze(S(:,j,:)),K)
#         nHit = nHit + Hitting
#         Pay[j] = payoff(iPayOff,np.squeeze(S(:,j,:)),Opt,Hitting)
#     
#     V = np.exp(- r * T) * mean(Pay) / Par
#     MsgBox(np.array(['Probability hitting barrier = ',num2str(nHit / nPath)]))
#     return V
#     
#     #==========================================================================
# #  hitting time down barrier
# #  returns 1 if the barrier is reached or 0 otherwise
#     
# def Hit(S = None,B = None): 
#     H = 0
#     for i in np.arange(1,S.shape[2-1]+1).reshape(-1):
#         if S(1,i) <= B:
#             H = 1
#     
#     return H
#     
#     return V


if __name__ == "__main__":
    # for testing purposes
    info = {
        "S0": np.array([8.7, 8.3]),
        "nPath": 10000,
        "nT": 1,
        "K" : 1,
        "T": 1,
        "Par": [1, 2],
        "iPayOff": 1,
        "payoff": payoff_basquet_call,
        'r': 0.01,
        'delta': np.array([[0], [0]]),
        "R": np.array([[1, 0.65], [0.65, 1]]),
        "D": np.array([0.23, 0.34]),
        "Sigma": np.array([[0.0529, 0.0508], [0.0508, 0.1156]]),
        }
    V = monte_carlo_pricing(iVC=1, **info)
    print(V)
