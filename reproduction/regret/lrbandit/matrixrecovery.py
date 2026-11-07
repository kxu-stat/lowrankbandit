# Extracted from kwangsungjun/lrbandit, commit 2f1f7ca4bbefe2bfd3e0bc50c4423a9791bfcde8
# (Apache License 2.0; see LICENSE).  `rankone` is matrixrecovery/matrixrecovery.py,
# `calc_sqrt_beta_det4` is blbandits3.py line 37.  Modifications are marked [MODIFIED].
import numpy as np, numpy.random as ra, scipy.linalg as sla
from numpy import log, sqrt


def calcRowwiseKron(D, X, ZV):
    """[MODIFIED] numpy version of matrixrecovery/myutils_cython.pyx:
    D[i, :] = np.outer(X[i, :], ZV[i, :]).ravel()"""
    D[:, :] = (X[:, :, None] * ZV[:, None, :]).reshape(D.shape)


def calc_sqrt_beta_det4(t, R, ridge, delta, Sp, logdetV, logdetV0):
    """ to allow diagonal regularization, actually I can have a better one.
    `Sp = sqrt(ridge) * S` in the previous versions.
    """
    return R * np.sqrt( logdetV - logdetV0 + log (1/(delta**2)) ) + Sp


def rankone(X,Z,y,r,R=.1, C=.1, tolPred=0.01, tolTh=0.01, maxIter=400, verbose=False, init_U=None):
    """
    matrix recovery with rank-one measurements using Burer-Monteiro approach 
    measurement model: (X[i,:] @ Theta) @ Z[i,:] == y[i]
    (IN)
      X, Z: N by d matrix
      y: N-dim vector
      r: the deemed rank of Theta
      R: noise level (subgaussian parameter)
      C: regularization parameter (larger => more regularization)
      tol: stopping condition
      maxIter: maximum number of iterations
      init_U: [MODIFIED] optional d by r start; default is the original ra.randn(d, r)
    (OUT)
      (U,V,out_nIter,stat) so that U@V.T ≈ Theta;
      stat['objs'] has the objective values over time
      stat['stoppingPredList'], stat['stoppingThetaList'] has stopping conditions over time
    """
    N,d = X.shape
    initU = ra.randn(d,r) if init_U is None else init_U     # [MODIFIED] seeded start
    U = initU
    V = initU # just a placeholder
    M = np.zeros( (d*r,d*r) )
    hatTh = initU @ initU.T # very bad initial hatTh
    my_tqdm = lambda x: x                                    # [MODIFIED] no progress bar

    objs = []; stoppingPredList = []; stoppingThetaList = []
    myeye = R*C*np.eye(d*r) 
    for iIter in my_tqdm(range(1,1+maxIter)):
        D = np.zeros((N,d*r))
        if iIter % 2 == 0: # update U
            ZV = Z @ V
            calcRowwiseKron(D, X, ZV) #- note D will be written!
        else: # update V
            XU = X @ U
            calcRowwiseKron(D, Z, XU)

        M[:,:] = myeye + D.T@D
        b = D.T @ y
        sol = sla.solve(M,b, assume_a='pos', overwrite_a=True).reshape(d,r)
        if iIter % 2 == 0:
            prevU = U
            U = sol
        else:
            prevV = V
            V = sol
        prev_hatTh = hatTh
        hatTh = U@V.T
        #- compute residual
        predy = ((X@hatTh)*Z).sum(1)
        obj = sla.norm(predy - y, 2)**2 + R*C*(sla.norm(U, 'fro')**2 + sla.norm(V, 'fro')**2)
        objs.append( obj )
        stoppingPred = sla.norm(predy - y, 2) / sla.norm(y,2)
        stoppingPredList.append( stoppingPred )
        stoppingTheta = sla.norm(hatTh - prev_hatTh, 'fro')
        stoppingThetaList.append( stoppingTheta )
        if (stoppingPred < tolPred):
            break
        if (stoppingTheta < tolTh):
            break
    out_nIter = iIter
    stat = {'objs': objs, 'stoppingPredList': stoppingPredList, 'stoppingThetaList': stoppingThetaList}
    return U,V,out_nIter,stat
