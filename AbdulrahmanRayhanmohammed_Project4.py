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

def spectral_radius(A):
    '''Returns maximum absolute eigenvalue for the givben matrix '''
    eigs = np.linalg.eig(A)[0]
    maxeig = max(abs(eigs))
    return maxeig



# Code for Project 4


# Function to find probability of a wavepacket

def prob_integral(array):
    '''This function provides the probability of finding a particle given an array corresponding to
    its discretized wavefunction. Please provide the array you're interested in.'''

    point_product = array * np.conj(array) # Multiplying by complex conjugate
    probability = np.sum(point_product) # Integral = sum for discrete values

    return np.real(probability)



# Schrodinger Equation Function

def sch_eqn(nspace, ntime, tau, method = 'ftcs', length = 200, potential = [], wparam = [10,0,0.5]):
    '''Function to solve 1 dimensional, time dependent Schroedinger Equation. Please provide the following:
    number of spatial grid points, number of time steps to be evolved, and the time step to be used. Optional arguments are:
    1. method: string, either ftcs or crank (Crank-Nicolson), 2. length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100),
    3. potential: 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. ,
    4. wparam: list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    
    Unit assumptions h_bar = 1 and m = 1/2. 

    Further information can be found in the code documentation; AbdulRahman_RayhanMohammed_project4.pdf'''

    # Making Appropriate grid for wavefunction called psi 

    x_vals = np.linspace(-length/2, length/2, nspace)
    t_vals = np.linspace(0, ntime * tau, ntime)
    psi = np.zeros((nspace,ntime), dtype=np.complex128)

    j = 1j # defining root of -1

    V = np.zeros(nspace) # Setting Potential
    for index in potential:
        V[index] = 1

    probabilities = np.zeros(ntime) # Setting probabiliy to zero

    # Using initial conditions from wparam
    sigma = wparam[0]
    x0 = wparam[1]
    k0 = wparam[2]
    

    #initial row using 9.42 from textbook
    init_row = np.e**(j*k0*x_vals) * np.e**(-((x_vals-x0)**2)/(2*(sigma**2))) / (np.sqrt(sigma*np.sqrt(np.pi)))
    init_row /= np.sqrt(prob_integral(init_row)) # Making sure initial condition is normalized
    probabilities[0] = (prob_integral(init_row)) # initial probability 
    psi[:,0] = init_row  
    I = make_tridiagonal(nspace,0,1,0) # identity matrix
    H = make_tridiagonal(nspace,-1,2,-1)
    np.fill_diagonal(H, np.diagonal(H) + V) # equation 9.31 in textbook with h_bar and m as 1 and 1/2

    # matrices for each method, from equations 9.32 (ftcs) and 9.40 (Crank Nicolson)
    ftcs_matrix = I - tau*j*H   
    crank_matrix_1 = np.linalg.inv(I + (j*tau/2 * H))
    crank_matrix_2 = I - (j*tau/2 * H) 

    # Checking if solution would be stable for ftcs
    calculate = True 
    if method.lower() == 'ftcs' and spectral_radius(ftcs_matrix) > 1:

        #print(spectral_radius(ftcs_matrix))
        calculate = False
        print('Warning, solution unstable since tau is too large. Integration not performed.')
        exit

    # Performing integration if solution expected to be stable
    if calculate == True:
        #print(spectral_radius(ftcs_matrix))
        for time in range(1,ntime):

            if method.lower() == 'ftcs':

                psi[:,time] = np.dot(ftcs_matrix,psi[:,time-1])
                probabilities[time] = prob_integral(psi[:,time])

            if method.lower() == 'crank':

                psi[:,time] = np.dot(crank_matrix_1 , np.dot(crank_matrix_2, psi[:, time - 1]))
                probabilities[time] = prob_integral(psi[:,time])
        
        return psi, x_vals, t_vals, probabilities

#print(sch_eqn(100,500,1e-5,'ftcs'))

# Plotting Function

def sch_plot(nspace, ntime, tau, plottype, specific_time, method = 'ftcs', length = 200, potential = [], wparam = [10,0,0.5],save='no'):
    '''Function that plots results for schrodinger equation. Please provide the following in order:
    number of spatial grid points, number of time steps to be evolved, the time step to be used, the plot you want (string) either
    psi( a plot of the real part of ψ(x) at a specific time t)  or prob (prob, a plot of the particle probability density ψ ψ*(x) at a specific time), and the specific
    time for when you want the plot.
     Optional arguments are:
    1. method: string, either ftcs or crank (Crank-Nicolson), 2. length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100),
    3. potential: 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. ,
    4. wparam: list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    4. save: string either yes or no if you want the plot to be saved. Default is no.
    
    Unit assumptions h_bar = 1 and m = 1/2. 

    Further information can be found in the code documentation; AbdulRahman_RayhanMohammed_project4.pdf'''
    
    # Using sch_eqn to find values for plotting
    results = sch_eqn(nspace,ntime,tau,method,length,potential,wparam)
    psi_matrix = results[0]
    x_vals = results[1]
    t_vals = results[2]
    probabilities = results[3]
    time_range = ntime * tau

    # Finding index of specific time in n time
    timeindex = None
    tolerance = 1e-5

    for timepoint in range(ntime):
        if abs(specific_time - timepoint * tau) < tolerance:
            timeindex = timepoint
            break
    
    if timeindex is None:
        print(f'Error: specific_time ({specific_time}) must be a multiple of time step tau ({tau}).')
        print(f'Valid time steps are multiples of {tau}, and should be within the time range {time_range}.')
        return
    
    if plottype.lower() == 'psi':

        real_at_t = np.real(psi_matrix[:,timeindex])
        plt.plot(x_vals, real_at_t)
        plt.show()

    if plottype.lower() == 'prob':

        prob_dens = psi_matrix[:,timeindex] * np.conj(psi_matrix[:,timeindex])
        plt.plot(x_vals, prob_dens)
        
        area2 = prob_integral(psi_matrix[:,timeindex])
        print(f"Area under the curve: {round(area2,3)}")
        plt.show()

    X, T = np.meshgrid(x_vals, t_vals)

    return results
ex = range(0,1000)

#sch_plot(100,900,1e-1,'psi',0,'crank',length=50)
sch_plot(500,9000,1e-1,'prob',800,'crank',length=50)

#increasing number of points in nspace decreases my probability somehow