# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 09:43:43 2015

@author: gabeo
adapted from Yu Hu's matlab scripts
"""

def tensor_aaa_d(A, B, C, d):
    '''
    three-tensor for three out-going matrix and a diagonal vector d    
    '''

    import numpy as np

    if d.ndim > 1: 
        raise Exception('diagonal d should be a vector in tensor_aaa_d')
    
    n = d.shape[0]   
#    na = A.shape[0]
#    nb = B.shape[0]
#    nc = C.shape[0]    
    
    t = np.zeros((n,n,n))
    for i in range(n):
        t[i,i,i] = d[i]
    
    t = tensor_M_T(A, t)
    
    t1 = np.transpose(t, (1,0,2))
    t1 = tensor_M_T(B, t1)
    t = np.transpose(t1, (1,0,2))
    
    t1 = np.transpose(t, (2,1,0))
    t1 = tensor_M_T(C, t1)
    t = np.transpose(t1, (2,1,0))
    
    return t
    
def tensor_M_T(M, T):
    '''
    tensor product between matrix and tensor    
    '''    
    import numpy as np    
    
    Tr = np.reshape(T, (T.shape[0], T.shape[1]*T.shape[2]))
    Tr = np.dot(M, Tr)
    A = np.reshape(Tr, (M.shape[0], T.shape[1], T.shape[2]))
    
    return A