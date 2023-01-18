import numpy as np


def geometric_brownian(S0, nPath, nT, T, Sigma, r, delta, D, antithetic=False,
                       *args, **kwargs): 
    """
    Generates a geometric brownian S, of size nPath x nT x len(S0).

    Parameters
    ----------
    
    Returns
    -------
    S: np.ndarray
    Geometric brownian
    Sa: np.ndarray
    Antithetic asset brownian
    """
    n_as = len(S0)
    S = np.zeros((nPath,n_as,nT + 1))
    S[:,:,0] = S0 * np.ones((nPath,1))
    if antithetic:
        Sa = np.copy(S)
    dX = np.zeros((nPath,n_as))
    sigL = np.linalg.cholesky(Sigma)
    drift = r - delta - D ** 2 / 2
    drift = drift[0] * np.ones((nPath,1))  # warning here, because we just get the first drift
    dt = T / nT
    sqrdt = np.sqrt(dt)
    np.random.seed(0)  # warning here too
    for i in np.arange(0,nT).reshape(-1):
        dX[:,:] = np.transpose(sigL.dot(np.random.randn(n_as,nPath)))
        qplus = np.exp(drift * dt + sqrdt * dX)
        S[:,:,i + 1] = np.multiply(S[:,:,i],qplus)
        if antithetic:
            qminus = np.exp(drift * dt - sqrdt * dX)
            Sa[:,:,i + 1] = np.multiply(Sa[:,:,i], qminus)
    if antithetic:
        return S, Sa
    return S


if __name__ == "__main__":
    # testing this geometric brownian works ~ unit test
    info = {
        "S0": np.array([8.7, 8.3]),
        "nPath": 10,
        "nT": 1,
        "K" : 1,
        "T": 1,
        "iPayOff": 1,
        'r': 0.01,
        'delta': np.array([[0], [0]]),
        "R": np.array([[1, 0.65], [0.65, 1]]),
        "D": np.array([0.23, 0.34]),
        "Sigma": np.array([[0.0529, 0.0508], [0.0508, 0.1156]]),
    }
    S = GeomBrownian(**info)
    print(S[:,:,-1])
