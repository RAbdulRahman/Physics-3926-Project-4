# Project 4 Physics 3926F

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


# Functions and code from older labs:

# Lab 10, Tridiagonal Matrix Maker, Gaussian Wavepacket Generator and Spectral Radius Calculator


def make_tridiagonal(N,b,d,a):

    '''Makes a symmetric matrix with diagonal and lines above and below diagonal.
    Provide size, values under diagonal, values for diagonal and values above diagonal.'''
    matrix = np.zeros((N,N))
    for i in range(N):
        matrix[i][i] = d #diagonal values
    for i in range(N-1):
        matrix[i][i+1] = a #above diagonal
        matrix[i+1][i] = b #below diagonal

    return matrix


def make_initialcond(sigma,k,grid):
    '''Makes initial conditions for wavepacket. Please provide sigma, k and grid size'''

    wavepacket = np.e**(-grid**2/(2*sigma**2))*np.cos(k*grid)

    return wavepacket


def spectral_radius(A):
    '''Returns maximum absolute eigenvalue for the givben matrix '''
    eigs = np.linalg.eig(A)[0]
    maxeig = max(abs(eigs))
    return maxeig


