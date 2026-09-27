
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#%% FUNCTIONS

def AMI(X,m,nb=16,plot=False, verbose=False, initial_verbose=True):
    """ 
    INPUT:
         x  - input time series
         m  - maximum delay parameter to consider
         nb - number of bins to use in calculation of the histogram (default = 16)
    OUTPUT:
         Ixy - average mutual information in nats (natural logarithm)
         tau - first minimum of the average mutual information (optimum delay)
    """ 
    
    if initial_verbose:
        print('\nEstimating embedding delay ...')
    
    if m < 3:
        raise ValueError('m must be at least 3 to search for a first minimum')

    # check maximum size possible
    n = len(X) - m

    x = np.array(X[0:n])

    Ixy = np.zeros((m,1)) 
    

   
    for k in range(m):
        y = np.array(X[k:n+k])
        Pxy = joint_prob(x,y,nb)
        # marginals from the joint histogram, so all three use the same bins
        Px = Pxy.sum(axis=1)
        Py = Pxy.sum(axis=0)
        
        for i in range(nb):
            for j in range(nb):
                if Pxy[i,j]==0:
                    continue
                else:
                    Ixy[k]+=Pxy[i,j] * (np.log(Pxy[i,j])-np.log(Px[i]*Py[j]))
    

    tau=1
    while ( Ixy[tau] > Ixy[tau+1] ) and (tau < m-2) :
        if tau == m-3:
            print('Warning! No first minimum in the delay specified. Going with maximum possible')
        tau += 1
    
    if verbose:
        print('The optimum delay is ',tau)
    
    if plot:
        
        plt.figure()
        plt.plot(np.arange(m),Ixy)
        plt.xlabel('Delay')
        plt.ylabel('Average Mutual Information')
        plt.show()

        
    return Ixy, tau

######################### SUBFUNCTIONS #######################################    
def marginal_prob(x,nb):
    """ 
    INPUT:
        x - time series
        nb   - number of bins for x
    OUTPUT: 
        Px - probability
    """ 
    #Px, edges = np.histogram(x, bins=nb, normed=True)       
    Px, edges = np.histogram(x, bins=nb)
    Px = Px / float(len(x))

    return Px
    

def joint_prob(x,y,nb):
    """ 
    INPUT:
        x & y - vectors (time series)
        nb   - number of bins for x
    OUTPUT: 
        Pxy - (nbx,nby) matrix
    """ 


    #Pxy, edgesx,edgesy  = np.histogram2d(x, y , bins=nb, normed=True)
    Pxy, edgesx,edgesy  = np.histogram2d(x, y , bins=nb)
    Pxy = Pxy / float(len(x))

    return  Pxy


#%% MAIN

if __name__=='__main__':
    
    from lorenz import Lorenz
    
    X = Lorenz(1000)  
    
    # example 1
    print('\n ---------------------- example ---------------------- ')
  
    m=100
    Ixy, tau = AMI(X,m,16,plot=True,verbose=True)
   