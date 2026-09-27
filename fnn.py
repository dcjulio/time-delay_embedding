import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


#%% FUNCTION

def false_nearest_neighbors(x, tau, D, thr, plot=False, verbose=False,initial_verbose=True):
    """ 
    INPUT:
        x  - scalar time series
        tau  - delay time (may use script ami.py)
        D  - maximum embedding dimension to consider (don't go crazy)
        thr - threshold (between 0 and 1) of points that aren't embedding the dimension (strongly recommend less than 0.01)
    OUTPUT:
        R  - false nearest neighbour for each dimension (optimum at zero)
        dim - embedding dimension given a threshold
    """
    if initial_verbose:
        #print '\nEstimating false nearest neighbors ... '
        print('\nEstimating false nearest neighbors ... ')    

    x = np.asarray(x, dtype=float)
    R = np.zeros((D,1))
    
    # check maximum size possible
    n = len(x) - D*tau 


    for k in range(1,D+1):
        
        if verbose:
            print('Estimating false nearest neighbors for dim ', k,'...')
        
        y = [[x[i+d*tau] for d in range(k)] for i in range(n)]
        tree = cKDTree(y)
        nn = tree.query(y,k=2)
        # with repeated values the tree may return the point itself as the second neighbour
        idx = [nn[1][t][1] if nn[1][t][0]==t else nn[1][t][0] for t in range(n)]
        dist = [nn[0][t][1] if nn[1][t][0]==t else nn[0][t][0] for t in range(n)]
        r = np.array([np.abs(x[idx[t]+(k)*tau]-x[t+(k)*tau])/dist[t]  for t in range(n)])
        R[k-1] = sum(r>10)/float(n)
        
    # calculates the embedding dimension given a threshold
    if R[-1] > thr:
        dim=D
        print('\nDimension does not converge with threshold', thr,'and took maximum possible. Lowest value is %.3f'% np.min(R))
    else:
        dim=np.where(R <= thr)[0][0] + 1
    
    if verbose:
        print('\n The optimum embedding dimension is ', dim)
        
    if plot:        
        plt.figure()
        plt.plot(range(1,D+1),R)
        plt.xlabel('Dimension')
        plt.ylabel('False nearest neighbours (fraction)')
        #plt.show()
        
    return R, dim
    
#%% MAIN

if __name__=='__main__':
	
    from lorenz import Lorenz

    X = Lorenz(1000)
    
    # example1
    print('\n ---------------------- example ---------------------- ')

    tau = 16 # delay
    D = 10 # maximum embedding dimension to consider
    thr = 0.01 # threshold

    
    R, dim = false_nearest_neighbors(X, tau, D, thr, plot=True, verbose=True)


    # plot
    plt.figure()
    plt.plot(X[:-tau], X[tau:], color='teal', markersize=2)
    plt.xlabel('$X_t$')
    plt.ylabel(f'$X_{{t+{tau}}}$')
    plt.title(f'2D Reconstruction (Embedding Dimension 2, $tau = {tau}$)')
    plt.grid(True, alpha=0.3)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Slicing for 3D: X(t), X(t+tau), X(t+2tau)
    x_axis = X[:-2*tau]
    y_axis = X[tau:-tau]
    z_axis = X[2*tau:]
    
    ax.plot(x_axis, y_axis, z_axis, color='darkorchid', lw=0.7, alpha=0.8)
    
    ax.set_xlabel('$X_t$')
    ax.set_ylabel(f'$X_{{t+{tau}}}$')
    ax.set_zlabel(f'$X_{{t+{2*tau}}}$')
    ax.set_title(f'3D Reconstruction (Embedding Dimension 3, $tau = {tau}$)')
    plt.show()