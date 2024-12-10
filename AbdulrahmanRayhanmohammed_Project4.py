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
    1. nspace: int, number of spatial grid points, 
    2. ntime: int, number of time steps to be evolved, 
    3. tau: float, the time step to be used. 
       The rest are optional and are:
    4. method: string, either ftcs or crank (Crank-Nicolson), 
    5. length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100),
    6. potential: 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. ,
    7. wparam: list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    
    Solver assumptions: h_bar = 1 and m = 1/2. 

    Further information can be found in the code documentation; AbdulRahman_RayhanMohammed_project4.pdf'''

    # Making Appropriate grid for wavefunction called psi 

    x_vals = np.linspace(-length/2, length/2, nspace)
    t_vals = np.linspace(0, ntime * tau, ntime)
    psi = np.zeros((nspace,ntime), dtype=np.complex128)

    j = 1j # defining root of -1 to reduce complexity of later lines of code

    V = np.zeros(nspace) # Setting Potential
    for index in potential:
        V[index] = 1

    probabilities = np.zeros(ntime) # Setting initial probabiliy to zero

    # Using initial conditions from wparam
    sigma = wparam[0]
    x0 = wparam[1]
    k0 = wparam[2]
    

    #initial row using 9.42 from textbook
    init_row = np.e**(j*k0*x_vals) * np.e**(-((x_vals-x0)**2)/(2*(sigma**2))) / (np.sqrt(sigma*np.sqrt(np.pi)))
    init_row /= np.sqrt(prob_integral(init_row)) # Making sure initial condition is normalized

    # Setting initial conditions and probability
    probabilities[0] = (prob_integral(init_row)) 
    psi[:,0] = init_row  

    # Creating matrices for calculation
    I = make_tridiagonal(nspace,0,1,0) # identity matrix
    H = make_tridiagonal(nspace,-1,2,-1)
    np.fill_diagonal(H, np.diagonal(H) + V) # equation 9.31 in textbook with h_bar and m as 1 and 1/2

    # matrices for each method, from equations 9.32 (ftcs) and 9.40 (Crank Nicolson)
    ftcs_matrix = I - tau*j*H   
    crank_matrix_1 = np.linalg.inv(I + (j*tau/2 * H))
    crank_matrix_2 = I - (j*tau/2 * H) 

    # Checking if solution would be stable for ftcs if selected method 
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
    1. nspace: int, number of spatial grid points, 
    2. ntime: int, number of time steps to be evolved, 
    3. tau: float, the time step to be used, 
    4. plottype: string, the plot you want, either psi( a plot of the real part of ψ(x) at a specific time t)  or prob (prob, a plot of the particle probability density ψ ψ*(x) at a specific time), 
    5. specific_time: float, the specific time for when you want the plot. Will be rounded to nearest multiple of tau if not initially an integer multiple of tau.
        The rest are optional arguments and are:
    6. method: string, either ftcs or crank (Crank-Nicolson), 2. length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100),
    7. potential: array, 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. ,
    8. wparam: list, list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    9. save: string, either yes or no if you want the plot to be saved. Default is no.
    
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
    tolerance = tau/2 # setting tolerance to find nearest multiple of tau

    for timepoint in range(ntime):
        if abs(specific_time - timepoint * tau) < tolerance:
            timeindex = timepoint
            break
    
    if timeindex is None:

        print(f'Valid time steps are multiples of {tau}, and should be within the time range {time_range}.')
        return
    
    if plottype.lower() == 'psi':

        real_at_t = np.real(psi_matrix[:,timeindex])
        # Plotting Real psi
        plt.plot(x_vals, real_at_t)
        plt.xlabel('x')
        plt.ylabel(f'Real part of ψ(x)')
        plt.title(f'Plot of the real part of ψ(x) at a time t = {specific_time}')

        plt.show()

    if plottype.lower() == 'prob':

        prob_dens = psi_matrix[:,timeindex] * np.conj(psi_matrix[:,timeindex])
        area = prob_integral(psi_matrix[:,timeindex]) # Recomputing probability just to show that it is conserved
        # Plotting Probability Density
        plt.plot(x_vals, prob_dens)
        plt.xlabel('x')
        plt.ylabel(f'Probability Density  ψ ψ*(x)')
        plt.title(f'Probability Density, ψ ψ*(x) at t = {specific_time}, Area Under Curve = {np.round(area,3)}')
        plt.show()


    ### Extra Code I Wrote To See How Probability Evolves ### Inspired By Lab 9

    # X, T = np.meshgrid( t_vals , x_vals)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, T, psi_matrix[:,:] * np.conj(psi_matrix[:,:]), cmap='viridis', edgecolor='none')
    # ax.set_ylabel('x position')
    # ax.set_xlabel('Time (s)')
    # ax.set_zlabel('Probability Density')
    # ax.set_title('3D Evolution of Probability Density')
    # plt.show()

    return 


ex = range(0,1000)

sch_plot(200,9000,1e-1,'prob',80.05,'crank',length=250)
#sch_plot(500,9000,1e-1,'prob',800.05,'crank',length=200)
