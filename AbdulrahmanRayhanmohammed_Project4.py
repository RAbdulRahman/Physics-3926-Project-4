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


# Code for Project 4

def sch_eqn(nspace, ntime, tau, method = 'ftcs', length = 200, potential = [], wparam = [10,0,0.5]):
    '''Function to solve 1 dimensional, time dependent Schroedinger Equation. Please provide the following:
    number of spatial grid points, number of time steps to be evolved, and the time step to be used. Optional arguments are:
    1. method: string, either ftcs or crank (Crank-Nicolson), 2. length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100),
    3. potential: 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. ,
    4. wparam: list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    
    Unit assumptions h_bar = 1 and m = 1/2. 

    Further information can be found in the code documentation; AbdulRahman_RayhanMohammed_project4.pdf'''

    # Making Appropriate grid for wavefunction called psi 

    x_vals = np.arange(-length/2, length/2, nspace)
    t_vals = np.arange(0,ntime,tau)
    psi = np.zeros((x_vals,t_vals))

    V = np.zeros(x_vals) # Setting Potential
    for index in potential:
        V[index] = 1

    probability = np.zeros(t_vals) # Setting probabiliy to zero

    # Using initial conditions from wparam
    sigma = wparam[0]
    x0 = wparam[1]
    t0 = wparam[2]
    k0 = 35 

    init_row = make_initialcond(sigma,k0,x_vals)
    psi[0,:] = init_row

    I = make_tridiagonal(nspace,0,1,0) # identity matrix
    H = make_tridiagonal(nspace,-1,2,-1) + V # equation 9.31 in textbook


    for time in range(ntime):

        if method.lower() == 'ftcs':



    return potential