# TODO complete this!
import numpy as np
import matplotlib as plt
from typing import Tuple, Optional
from laplacian_pyramid import rowdec, rowdec2, rowint, rowint2
from laplacian_pyramid import quant1, quant2
from laplacian_pyramid import bpp

__all__ = ["h1", "h2", "g1", "g2", "dwt", "idwt"]

h1 = np.array([-1, 2, 6, 2, -1])/8
h2 = np.array([-1, 2, -1])/4

g1 = np.array([1, 2, 1])/2
g2 = np.array([-1, -2, 6, -2, -1])/4


def dwt(X: np.ndarray, h1: np.ndarray = h1, h2: np.ndarray = h2) -> np.ndarray:
    """
    Return a 1-level 2-D discrete wavelet transform of X.

    Default h1 and h2 are the LeGall filter pair.

    Parameters:
        X: Image matrix (Usually 256x256)
        h1, h2: Filter coefficients
    Returns:
        Y: 1-level 2D DWT of X
    """
    m, n = X.shape
    if m % 2 or n % 2:
        raise ValueError("Image dimensions must be even")
    Y = np.concatenate([rowdec(X, h1), rowdec2(X, h2)], axis=1)
    Y = np.concatenate([rowdec(Y.T, h1).T, rowdec2(Y.T, h2).T], axis=0)
    return Y


def idwt(X: np.ndarray, g1: np.ndarray = g1, g2: np.ndarray = g2)-> np.ndarray:
    """
    Return a 1-level 2-D inverse discrete wavelet transform on X.

    If filters G1 and G2 are given, then they are used, otherwise the LeGall
    filter pair are used.
    """
    m, n = X.shape
    if m % 2 or n % 2:
        raise ValueError("Image dimensions must be even")
    m2 = m//2
    n2 = n//2
    Y = rowint(X[:m2, :].T, g1).T + rowint2(X[m2:, :].T,g2).T;
    Y = rowint(Y[:, :n2], g1) + rowint2(Y[:, n2:], g2)
    return Y

def nlevdwt(X, n):
    # your code here
    m = 256
    Y = dwt(X)
    

    for i in range(n):
        m=m//2
        Y[:m,:m] = dwt(Y[:m,:m])

    return (Y)

def nlevidwt(Y, n):
    m = int(256 * 0.5**(n))
    # print(m)
    for i in range(n+1):
        Y[:m,:m] = idwt(Y[:m,:m])
        m = m*2
        # plt.imshow(Y,cmap = 'gray')
        # plt.show()
    return(Y)

def quantdwt1(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    n = np.shape(dwtstep)[1] 
    # print(n)
    m = 256
    dwtbits = np.empty((3,n))
    dwtents = np.empty((3,n))
    Yq = np.zeros((256,256))
    
    for i in range(n-1):
        m = m//2
        
        Y00 = quant1(Y[m:2*m,:m], dwtstep[0,i], rise1=dwtstep[0,i])
        Y10 = quant1(Y[m:2*m,m:2*m], dwtstep[1,i], rise1=dwtstep[1,i])
        Y20 = quant1(Y[:m,m:2*m], dwtstep[2,i], rise1=dwtstep[2,i])
        
        EY00 = bpp(Y00)*m*m
        EY10 = bpp(Y10)*m*m
        EY20 = bpp(Y20)*m*m
        
        EY00_e = bpp(Y00)
        EY10_e = bpp(Y10)
        EY20_e = bpp(Y20)

        dwtents[0][i] = EY00
        dwtents[1][i] = EY10
        dwtents[2][i] = EY20
        
        dwtbits[0][i] = EY00_e
        dwtbits[1][i] = EY10_e
        dwtbits[2][i] = EY20_e

        Yq[m:2*m,:m] = Y00
        Yq[m:2*m,m:2*m] = Y10
        Yq[:m,m:2*m] = Y20
        
    # m = m//2
    
    Y03 = quant1(Y[:m,:m], dwtstep[0,0])
    Yq[:m,:m] = Y03
    EY03 = bpp(Y03)*m*m
    EY03_e = bpp(Y03)

    return Yq, dwtbits, dwtents, EY03 , EY03_e

def quantdwt2(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    n = np.shape(dwtstep)[1] 
    # print(n)
    m = 256
    dwtbits = np.empty((3,n))
    dwtents = np.empty((3,n))
    Yq = np.zeros((256,256))
    
    for i in range(n-1):
        m = m//2
        
        Y00 = quant2(Y[m:2*m,:m], dwtstep[0,i], rise1=dwtstep[0,i])
        Y10 = quant2(Y[m:2*m,m:2*m], dwtstep[1,i], rise1=dwtstep[1,i])
        Y20 = quant2(Y[:m,m:2*m], dwtstep[2,i], rise1=dwtstep[2,i])
        
        EY00 = bpp(Y00)*m*m
        EY10 = bpp(Y10)*m*m
        EY20 = bpp(Y20)*m*m
        
        EY00_e = bpp(Y00)
        EY10_e = bpp(Y10)
        EY20_e = bpp(Y20)

        dwtents[0][i] = EY00
        dwtents[1][i] = EY10
        dwtents[2][i] = EY20
        
        dwtbits[0][i] = EY00_e
        dwtbits[1][i] = EY10_e
        dwtbits[2][i] = EY20_e

        Yq[m:2*m,:m] = Y00
        Yq[m:2*m,m:2*m] = Y10
        Yq[:m,m:2*m] = Y20
        
    # m = m//2
    
    Y03 = quant2(Y[:m,:m], dwtstep[0,0])
    Yq[:m,:m] = Y03
    EY03 = bpp(Y03)*m*m
    EY03_e = bpp(Y03)

    return Yq, dwtbits, dwtents, EY03 , EY03_e

def dwtImpulseResponse(levels):
    X_dummy = np.zeros((256,256))
    Y_dummy = nlevdwt(X_dummy,levels)
    m = 256
    #plt.imshow(Y_dummy,cmap = 'gray')
    #plt.show()
    sub_images = []
    energy_out =[]
    layer_centers = []

    for i in range(levels):
        m = m//2
        layer_centers.append((m + m//2,m//2))
        layer_centers.append((m + m//2,m+m//2))
        layer_centers.append((m//2,m +m//2))

    for x in (layer_centers):
        Y_dummy_copy = Y_dummy.copy()
        
        
        Y_dummy_copy[x[0],x[1]] = 100
        #Reconstruct
        Z = nlevidwt(Y_dummy_copy,levels)
        energy_out.append(np.sum(Z**2))
        
        # plt.imshow(Z,cmap = 'gray')
        # plt.show()
        
    final_center = (m//2,m//2)
    Y_dummy_copy = Y_dummy
    Y_dummy_copy[final_center] = 100
    Z = nlevidwt(Y_dummy_copy,levels)
    energy_out.append(np.sum(Z**2))

    return energy_out


def dwtQuantMatrix(qstep, levels, energies: np.array = np.zeros((256, 256)), EqualStep: bool = True, EqualMse: bool = False):

    if EqualStep == 1:
        dwtstep = np.empty((3, levels))
        for x in range(levels):
            dwtstep[0][x] = qstep
            dwtstep[1][x] = qstep
            dwtstep[2][x] = qstep

        return dwtstep

    elif EqualMse == 1:
        dwtstep = np.zeros((3,levels+1))
        steps = []
        for i in energies:
            steps.append(qstep*(energies[0]/i))
        for i in range(0, levels*3,3):
            # print(i//3)
            dwtstep[0,i//3] = steps[i]
            dwtstep[1,i//3] = steps[i+1]
            dwtstep[2,i//3] = steps[i+2]
            col = i//3
        dwtstep[0,col+1] = steps[-1]

        return dwtstep
    
    else:
        "Error: one of constant step and eqila mse must be true"