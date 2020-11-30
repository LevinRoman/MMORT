import pickle
import pathlib
import os
import scipy.linalg
import scipy.sparse
import scipy.spatial
import numpy as np

def save_obj(obj, name, directory = ''):
    pathlib.Path(os.path.join('obj', directory)).mkdir(exist_ok=True)
    with open(os.path.join('obj', directory, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, directory = ''):
    with open(os.path.join('obj', directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
    
    
    
def construct_beam_neighbor_difference_matrix(beam, eps = 5):
    """Compute a smoothing matrix. (Linear constraint: abs diff between neighbors <= k)
    
    Parameters
    ----------
    beam : np.array of shape (num_beamlets,None)
        beamlet positions, usually in 2D or 3D
    
    eps : float
        radius in which to look for neighbors
        
    Returns
    -------
    sparse matrix 
        neighbor difference matrix for a beam
        
    Notes
    -----
    What is the deliverability condition to test?
    """
    print(beam.shape)
    tol = eps*0.01 #tolerance for inexact coordinates
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(beam))
    
    neighbors = (distances <= eps+tol) & (distances > tol)

    neighbor_difference_beam = []
    for i in range(neighbors.shape[0]):
        for j in range(i, neighbors.shape[1]): #walk through the upper triangle of the distance matrix
            if neighbors[i,j]: #if u_i and u_j are neighbors, add their difference, the order does not matter, will be square
                neighbor_difference = np.zeros(beam.shape[0])
                neighbor_difference[i] = 1
                neighbor_difference[j] = -1
                neighbor_difference_beam.append(neighbor_difference)
    #Multiply by 1/eps to get the finite difference-ish
    return (1/eps)*np.array(neighbor_difference_beam)


def construct_smoothing_matrix(beams, eps = 5):
    """Compute a smoothing matrix. (Linear constraint: abs diff between neighbors <= k)
    
    Parameters
    ----------
    beams : list of np.arrays of shape (num_beamlets[i],None) for i-th beam
        beams with beamlet positions, usually in 2D or 3D
    eps : float
        radius in which to look for neighbors
        
    Returns
    -------
    sparse block diag matrix 
        Smoothing matrix S
    """
    S_array = [construct_beam_neighbor_difference_matrix(beam, eps = eps) for beam in beams]
    S = scipy.sparse.block_diag(S_array, format='csr') #don't need sparse here, we are doing dense A in u_update
#     S = scipy.linalg.block_diag(S_array)
    return S

import os
import csv


def save_output_from_dict(out_dir, state, file_name):    # Read input information
    args = []
    values = []
    for arg, value in state.items():
        args.append(arg)
        values.append(value)
        # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, file_name)
    fieldnames = [arg for arg in args]
    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except:
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            # Add row for this experiment
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow({arg: value for (arg, value) in zip(args, values)})
    print('\nResults saved to '+fname+'.')