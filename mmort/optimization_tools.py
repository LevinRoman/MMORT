import numpy as np
import scipy.optimize
import pandas as pd
import scipy.io
import scipy
import scipy.sparse
import time


#Helper functions
def prox(eta_0, y, B):
    """Prox_{-eta_0 f}(y) = argmin_x {-f(x) + \frac{1}{2\eta_0}\|x-y\|^2} where 
    f = x^T B x,
    B = block_diag(N_m diag(\beta_m)) for m = 1...M --- modalities
    Based on the paper "Proof of Principle: Multi-Modality Radiotherapy Optimization", section 3.2.1
    Used in the update of the auxilary variable w_0

    Parameters
    ----------
    eta_0 : float
        The tumor penalty
    y : np.array of shape (None,)
        The point at which to evaluate the prox
    B  : np.array of shape (None, )
        The block-diagonal coefficient array B since the matrix in f is assumed to be diagonal
        (The quadratic coefficient array of the tumor BE with N_m included as coeffs)

    Returns
    -------
    prox : np.array of shape (None,)
        Returns the prox value at y 
        
    Raises
    ------
    ValueError
        If the prox does not exist per section 3.2.1
        (When np.max(B - (1/(2*eta_0))) > 0)

    """
    #Existence check, see the paper, section 3.2.1
    if np.max(B - (1/(2*eta_0))) > 0:
        raise ValueError('Prox does not exist!')
    prox = ((1/eta_0)*y)/(-2*B + (1/eta_0))
    return prox


#C - constraint, D - array, dont forget np.diag when multiplying
def proj(v, gamma, D, C):
    """proj_{\Omega}(v) = \arg\min_{w \in \Omega} \|w - v\|^2  where
    \Omega = \{w: \tilde \gamma^T w + w^T D w \le C\}
    
    In the paper indices i are used for i-th generalized OAR:
    proj_{\Omega_i}(v) = \arg\min_{w \in \Omega_i} \|w - v\|^2  where
    \Omega_i = \{w_i: \tilde {\gamma^i}^T w_i+w_i^TD^iw_i \le C^i\}
    
    D^i = block_diag(N_m diag(\delta_m^i)) for m = 1...M -- modalities, i -- generalized OARs
    \tilde \gamma^i = [N_m \gamma^i_m] for m -- modalities, i -- generalized OARs
    Based on the paper "Proof of Principle: Multi-Modality Radiotherapy Optimization", section 3.2.1 and Appendix
    Used in the update of the auxilary variables w_i, i = 1, 2, ... for the i-th generalized OAR 
    (Generalized OARs are mean-dose OARs + all voxels of max-dose OARs considered as separate OARs)

    Parameters
    ----------
    v : np.array of shape(None)
        The point to project onto \Omega 
    gamma : np.array of shape (None,)
        The linear block coefficient array of the OAR BE (with N_m included as coefficients)
    D : np.array of shape (None, )
        The quadratic block-diagonal coefficient array D of the OAR BE
        (since the matrix in \Omega is assumed to be diagonal)
    C : float
        The constraint value 
        (Should be differently computed for mean and max dose OARs fromt the input data)
        
    Returns
    -------
    proj : np.array of shape (None,)
        Returns the proj of v onto \Omega
        
    Raises
    ------
    ZeroDivisionError
        If some of the used matrices are not invertible and there is division by zero!
        
    Notes
    -----
    Changes: np.diag(some_array)@l changed to some_array*l
             changed G notation to D notation as per the paper
        
    """
    D_hat = 1/(np.sqrt(D))
    l = gamma/(2*D) + v
    K = C + np.sum((0.5*gamma/np.sqrt(D))**2)
    w = lambda z: ((1/np.sqrt(D))*(z - 0.5*(gamma/np.sqrt(D)))) #go back to w
    #check the least squares solution:
    z_ls = (1/D_hat)*l
    if np.linalg.norm(z_ls,2)**2 <= K:
        proj = w(z_ls)
        return proj
    #Otherwise, look for Lambda
    obj = lambda Lambda: ((np.linalg.norm(((1/(D_hat**2 + Lambda*np.ones(len(D_hat))))*D_hat)*l,2))**2 - K)
    Lambda_opt = scipy.optimize.fsolve(obj, 0)[0]
    z_opt = ((1/(D_hat**2 + Lambda_opt*np.ones(len(D_hat))))*D_hat)*l
    proj = w(z_opt)
    if np.sum(np.isnan(proj)) > 0:
        raise ZeroDivisionError('Division by zero! Some matrices are not invertible!')
    return proj

def obj_u_opt_N_fixed(u, T, alpha, B):
    """Objective function without tumor doubling and with N in coefficients.
    Based on the paper "Proof of Principle: Multi-Modality Radiotherapy Optimization",
    right before the section 3.2.1

    Parameters
    ----------
    u : np.array of shape(None,)
        Beamlet distribution 
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix 
        block_diag(T_1, ... , T_M) where T_i is the sparse dose deposition matrix for i-th modality
    alpha : np.array of shape (None,)
        The linear block coefficient array of the tumor BE 
        (with N_m included as coefficients)
    B : np.array of shape (None,)
        The quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m included as coefficients)
   
    Returns
    -------
    float
        Objective value
        
    Notes
    -----
    Changes: np.diag(some_array)@another_array changed to some_array*another_array
             changed Beta notation to B notation as per the paper
             changed A notation to T notation as per the paper
             changed the order of the arguments to more logical
             changed to the possibly sparse T interface, @ -> T.dot
        
    """
    x = T.dot(u)
    return alpha.T.dot(x) - x.T.dot(B*x)

def relaxed_obj_u_opt_N_fixed(u, w_0, w, eta_0, eta, T, H, alpha, B):
    """Relaxed objective function without tumor doubling and with N in coefficients.
    Based on the paper "Proof of Principle: Multi-Modality Radiotherapy Optimization",
    section 3.2.1

    Parameters
    ----------
    u : np.array of shape(None,)
        Beamlet distribution 
    w_0 : np.array of shape (None,)
        Auxilary dose distribution, optimal for the tumor
    w : List of np.arrays of shape (None,)
        Auxilary dose distributions feasible for OARs
    eta_0 : float
        Penalty parameter for the tumor
    eta : np.array of shape (None,)
        Array of OAR penalty parameters
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix 
        block_diag(T_1, ... , T_M) where T_i is the sparse dose deposition matrix for i-th modality
    H : List of sparse arrays of shape(None, None)
        List of block-diagonal OAR dose-deposition matrix block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality 
        (each element of the list corresponds to a generalized OAR)
    alpha : np.array of shape (None,)
        The linear block coefficient array of the tumor BE 
        (with N_m included as coefficients)
    B : np.array of shape (None,)
        The quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m included as coefficients)
   
    Returns
    -------
    float
        Objective value
        
        
    """
    tumor_dose = T.dot(u)
    OAR_doses = [Hi.dot(u) for Hi in H]
    relaxed_obj = alpha.T.dot(w_0) - w_0.T.dot(B*w_0) + (1/(2*eta_0))*(np.linalg.norm(
        w_0-tumor_dose))**2 + np.sum([(1/(2*eta[i]))*(np.linalg.norm(w[i]-OAR_doses[i]))**2 for i in range(len(H))])
    return relaxed_obj

#############################################################################################
#Review this one, it is not a function of N!
def obj_u_opt_N_opt(u, T, alpha, B, N, num_tumor_voxels, Td = 2):
    """Objective function with N in tumor doubling and with N implicit in modality coefficients.
    Based on the paper "Proof of Principle: Multi-Modality Radiotherapy Optimization",
    the general objective function in (P0).
    Used for optimization over fractionation schedule N.

    Parameters
    ----------
    u : np.array of shape(None,)
        Beamlet distribution 
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix block_diag(T_1, ... , T_M) 
        where T_i is the sparse dose deposition matrix for i-th modality
    alpha : np.array of shape (None,)
        Linear block coefficient array of the tumor BE 
        (with N_m NOT included as coefficients)
    B : np.array of shape (None,)
        Quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m NOT included as coefficients)
    N : np.array of shape (None,)
        Fractionation schedule for M modalities 
    num_tumor_voxels : int
        Number of tumor voxels (in the future could be different for different modalities in case of uncertainty)
    Td : int
        Tumor doubling time in days
   
    Returns
    -------
    float
        Objective value
        
    Notes
    -----
    Changes: np.diag(some_array)@another_array changed to some_array*another_array
             changed Beta notation to B notation as per the paper
             changed A notation to T notation as per the paper
             changed the order of the arguments to more logical
             changed to the possibly sparse T interface, @ -> T.dot
        
    """
    x = T.dot(u)
    alpha_tilde = alpha #np.repeat(N, num_tumor_voxels)*alpha
    B_tilde = B #np.repeat(N, num_tumor_voxels)*B
    #Note that all modalities must have the same number of tumor voxels:
    return alpha_tilde.T.dot(x) - x.T.dot(B_tilde*x) + num_tumor_voxels*(np.sum(N)-1)*(np.log(2)/Td)
###########################################################################################################

def constraint(u, H, gamma, D, C, tol, verbose = 0):
    """Check the constraint violation for given u, fixed N

    Parameters
    ----------
    u : np.array of shape(None,)
        Beamlet radiation distribution 
    H : Sparse array of shape(None, None)
        Block-diagonal OAR dose-deposition matrix block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality (for a given OAR)
    gamma : np.array of shape (None,)
        Linear block coefficient array of the OAR BE 
        (with N_m included as coefficients, vector gamma_tilde in the paper)
    D : np.array of shape (None,)
        Quadratic block-diagonal coefficient array D of the OAR BE
        (with N_m included as coefficients, D as it is in the paper, currently the notation is not ideal)
    C : float
        Constraint to satisfy
    tol : float
        Tolerance to check the soft constraint violation
    verbose : 0 or 1
        0 for no printing
        1 for printing
   
    Returns
    -------
    tuple (bool, float, float, float)
        (soft_constr_satisfied, distance, constr_val at u, C)
        
    Notes
    -----
    Changes: np.diag(some_array)@another_array changed to some_array*another_array
             changed G notation to D notation as per the paper
             changed B notation to H notation as per the paper
             changed the order of the arguments to more logical
             changed to the possibly sparse H interface, @ -> H.dot
        
    """
    x = H.dot(u)
    constr_val = gamma.T.dot(x) + x.T.dot(D*x)
    hard = constr_val <= C
    soft = constr_val <= (1+tol)*C
    if bool(verbose):
        print('   Hard constraint satisfied: {}'.format(hard))
        print('   Relaxed constraint within {}% tol satisfied: {}'.format(tol*100, soft))
    return soft, 100*(constr_val - C)/C, constr_val, C

def constraints_all(u, H, gamma, D, C, tol = 0.05, verbose = 0):
    """Check the constraint violation for given u, fixed N

    Parameters
    ----------
    u : np.array of shape(None,)
        Beamlet radiation distribution 
    H : List of sparse arrays of shape(None, None)
        List of block-diagonal OAR dose-deposition matrix block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality 
        (each element of the list corresponds to a generalized OAR)
    gamma : np.array of shape (None,)
        List of linear block coefficient arrays of the OAR BE 
        (with N_m included as coefficients, vector gamma_tilde in the paper)
        (each element of the list corresponds to a generalized OAR)
    D : List of np.arrays of shape (None,)
        Quadratic block-diagonal coefficient array D of the OAR BE
        (with N_m included as coefficients, D as it is in the paper, currently the notation is not ideal)
    C : array-like of floats
        Constraints to satisfy
    tol : float
        Tolerance to check the soft constraint violation
    verbose : 0 or 1
        0 for no printing
        1 for printing
   
    Returns
    -------
    constraints : pd.DataFrame with len(C) rows of (bool, float, float, float)
        Each row correponds to a constraint and is (soft_constr_satisfied, distance, constr_val at u, C)
        
    """
    constraints = [constraint(u, H[i], gamma[i], D[i], C[i], tol, verbose) for i in range(len(H))]
    constraints = pd.DataFrame(constraints, columns = ['Relaxed', 'Distance', 'Constr at u_opt', 'actual constr'])
    return constraints



#update in u:
#The following is an attempt to incorporate smoothing
# def u_update(eta_s, eta_0, eta, w_s, w_0, w, eta_S_T_H_stacked, premultiplied_lhs = None, nnls_max_iter=50):
#     """Compute the sparse least squares update for u per section 3.2.1 of the paper
#     The rhs of the ls problem needs to be recomputed every time since w_0 and w are variables
#     The lhs should be precomputed in advance for computational efficiency

#     Parameters
#     ----------
#     eta_0 : float
#         Penalty parameter for the tumor
#     eta : np.array of shape (None,)
#         Array of OAR penalty parameters
#     w_0 : np.array of shape (None,)
#         Auxilary dose distribution, optimal for the tumor
#     w : List of np.arrays of shape (None,)
#         Auxilary dose distributions feasible for OARs
#     eta_T_H_stacked : Sparse array of shape (None, None)
#         Stacked dose deposition matrices, premultiplied by penalty parameters, needs to be precomputed in the solver, 
#         o.w. expensive to do at every iteration, lhs of the least squares update
#     premultiplied_lhs : Dense array of shape (None, None)
#         A.T@A for the nnls, old solver
#     nnls_max_iter : int
#         Max iterations for the internal scipy nnls
        
#     Returns
#     -------
#     u_next : np.array of shape (None,)
#         update for the beamlet radiation distribution u
        
#     Notes
#     -----
#     Changes: Took the lhs computation out of the update to make it more efficient
#     TODO: Add a check for sparsity? Fix the hack with [[]], add the option to choose solvers
#     """
#     """In the following  +[[]] and [:-1] are added to keep 1dim array of objects and still multiply it elemtwisely"""
#     w_concat = np.concatenate((1/np.sqrt(2*eta))*np.array(w+[[]])[:-1], axis = 0) #[:-1] Added as a hack to keep it one-dim array of objects
#     b_ls = np.concatenate([(1/np.sqrt(2*eta_s))*w_s, (1/np.sqrt(2*eta_0))*w_0, w_concat], axis = 0)
#     u_next = scipy.optimize.lsq_linear(eta_S_T_H_stacked, b_ls, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-1, max_iter=nnls_max_iter, verbose=1).x
#     return u_next



# def u_update(eta_0, eta, w_0, w, eta_T_H_stacked, premultiplied_lhs = None, nnls_max_iter=50):  
#     # PREMULTIPLIED LHS IS AN EXTRA ARGUMENT! Set it to None and add solver!    
#     """Compute the sparse least squares update for u per section 3.2.1 of the paper 
#     The rhs of the ls problem needs to be recomputed every time since w_0 and w are variables   
#     The lhs should be precomputed in advance for computational efficiency   
    
#     Parameters  
#     ----------  
#     eta_0 : float   
#         Penalty parameter for the tumor 
#     eta : np.array of shape (None,) 
#         Array of OAR penalty parameters 
#     w_0 : np.array of shape (None,) 
#         Auxilary dose distribution, optimal for the tumor   
#     w : List of np.arrays of shape (None,)  
#         Auxilary dose distributions feasible for OARs   
#     eta_T_H_stacked : Sparse array of shape (None, None)    
#         Stacked dose deposition matrices, premultiplied by penalty parameters, needs to be precomputed in the solver,   
#         o.w. expensive to do at every iteration, lhs of the least squares update    
#     premultiplied_lhs : Dense array of shape (None, None)   
#         A.T@A for the nnls, old solver  
#     nnls_max_iter : int 
#         Max iterations for the internal scipy nnls  
            
#     Returns 
#     ------- 
#     u_next : np.array of shape (None,)  
#         update for the beamlet radiation distribution u 
            
#     Notes   
#     -----   
#     Changes: Took the lhs computation out of the update to make it more efficient   
#     TODO: Add a check for sparsity? Fix the hack with [[]], add the option to choose solvers    
#     Another idea: use u = nnls_predotted(premultiplied_lhs, premultiplied_rhs, tol=1e-8)    
#                   with premultiplied_lhs precomputed outside the solver loop, then uncomment the nnls   
#                   (create a separate file with it)  
#     """ 
#     """In the following  +[[]] and [:-1] are added to keep thing 1dim array of objects and still multiply it elemtwisely""" 
#     import time

#     start = time.time()


# #     #B.append([]) #THIS IS WRONG, CHANGES THE LIST    
# #     B_concat = np.concatenate((1/np.sqrt(2*eta))*np.array(B+[[]])[:-1], axis = 0) 
# #     A_ls = np.concatenate([(1/np.sqrt(2*eta0))*A, B_concat], axis = 0)    
# #     #print(np.array(B).shape) 
# #     #print(w[0].shape)    
# #     #print(w, eta)    
# #     #w.append([]) THIS IS WRONG, CHANGES THE LIST 
# #     w_concat = np.concatenate((1/np.sqrt(2*eta))*np.array(w+[[]])[:-1], axis = 0) #[:-1] Added as a hack to keep it one-dim array of objects  
# #     eta_w = np.expand_dims(1/np.sqrt(2*eta),1)*np.array(w)    
# #     print(eta_w.shape)    
# #     b_ls = np.concatenate([(1/np.sqrt(2*eta_0))*w_0, eta_w.flatten()], axis = 0)  
#     w_concat = np.concatenate((1/np.sqrt(2*eta))*np.array(w+[[]])[:-1], axis = 0) #[:-1] Added as a hack to keep it one-dim array of objects    
#     b_ls = np.concatenate([(1/np.sqrt(2*eta_0))*w_0, w_concat], axis = 0)   
# #     print(np.sum(eta_w.flatten() != w_concat))    
# #     premultiplied_time_start = time.time()    
# #     premultiplied_lhs = eta_T_H_stacked.T.dot(eta_T_H_stacked).toarray()  
# #     premultiplied_time_end = time.time()  
# #     print('premultiplying took {}'.format(premultiplied_time_end - premultiplied_time_start)) 
# #     premultiplied_rhs = eta_T_H_stacked.T.dot(b_ls)   
# #     u_next = nnls_predotted(premultiplied_lhs, premultiplied_rhs, tol=1e-5)   
# #     print(eta_T_H_stacked.shape, b_ls.shape)  
# #     A_ls_t_b = eta_T_H_stacked.T.dot(b_ls)    
# #     w =scipy.sparse.linalg.spsolve_triangular(RT, A_ls_t_b, lower = True) 
# #     x = scipy.sparse.linalg.spsolve_triangular(R, w, lower = False)   
# #     u_next = x    
# #     scipy.sparse.save_npz("nnls_LHS.npz", eta_T_H_stacked)
# #     scipy.sparse.save_npz("nnls_RHS.npz", b_ls)
#     u_next = scipy.optimize.lsq_linear(eta_T_H_stacked, b_ls, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-1, max_iter=nnls_max_iter, verbose=1).x   
# #     u = scipy.optimize.lsq_linear(premultiplied_lhs, premultiplied_rhs, bounds = (0, np.inf), tol=1e-5).x 
#     end = time.time()
#     print('u update took:', end - start)
#     return u_next


def u_update(u_cur, AtA, AA, S, StS, lambda_smoothing, eta_0, eta, w_0, w, eta_T_H_stacked, nnls_max_iter=100):  
    # PREMULTIPLIED LHS IS AN EXTRA ARGUMENT! Set it to None and add solver!    
    """Compute the sparse least squares update for u per section 3.2.1 of the paper 
    The rhs of the ls problem needs to be recomputed every time since w_0 and w are variables   
    The lhs should be precomputed in advance for computational efficiency   
    
    Parameters  
    ----------  
    u_cur : np.array
        Current u vector
    AtA : sparse matrix
        LHS of the normal equations of the least squares with eta_T_H_stacked
    AA : np.array
        Dense LHS of the normal equations of the least squares with eta_T_H_stacked
    eta_0 : float   
        Penalty parameter for the tumor 
    eta : np.array of shape (None,) 
        Array of OAR penalty parameters 
    w_0 : np.array of shape (None,) 
        Auxilary dose distribution, optimal for the tumor   
    w : List of np.arrays of shape (None,)  
        Auxilary dose distributions feasible for OARs   
    eta_T_H_stacked : Sparse array of shape (None, None)    
        Stacked dose deposition matrices, premultiplied by penalty parameters, needs to be precomputed in the solver,   
        o.w. expensive to do at every iteration, lhs of the least squares update    
    premultiplied_lhs : Dense array of shape (None, None)   
        A.T@A for the nnls, old solver  
    nnls_max_iter : int 
        Max iterations for the internal scipy nnls  
            
    Returns 
    ------- 
    u_next : np.array of shape (None,)  
        update for the beamlet radiation distribution u 
            
    Notes   
    -----   
    Changes: Took the lhs computation out of the update to make it more efficient   
    TODO: Add a check for sparsity? Fix the hack with [[]], add the option to choose solvers    
    Another idea: use u = nnls_predotted(premultiplied_lhs, premultiplied_rhs, tol=1e-8)    
                  with premultiplied_lhs precomputed outside the solver loop, then uncomment the nnls   
                  (create a separate file with it)  
    """ 
    """In the following  +[[]] and [:-1] are added to keep the thing 1dim array of objects and still multiply it elemtwisely""" 
    import time

    start = time.time()
    w_concat = np.concatenate((1/np.sqrt(2*eta))*np.array(w+[[]])[:-1], axis = 0) #[:-1] Added as a hack to keep it one-dim array of objects    
    b_ls = np.concatenate([(1/np.sqrt(2*eta_0))*w_0, w_concat], axis = 0)   
    

    # from numdifftools import Jacobian, Hessian
    #This part is wrong in terms of protons, will still do smoothin erroneously
    def fun(x, A, b, AtA, Atb, S, StS, lambda_smoothing):
        return (1/2)*np.linalg.norm(A.dot(x) - b)**2 + (lambda_smoothing/2)*np.linalg.norm(S.dot(x[:S.shape[1]]))**2

    def grad(x, A, b, AtA, Atb, S, StS, lambda_smoothing):
        # AA = np.ones((10,10))
        # AA[:5,:5] = AA[:5,:5]+1
        # AA
        #S corresponds only to the first modality, so adding that to a block of A
        photon_shape = StS.shape[0]
        AtA[:photon_shape, :photon_shape] = AtA[:photon_shape, :photon_shape] + lambda_smoothing*StS
        return AtA.dot(x) - Atb 

#     def hess(x, A, b, AtA, Atb):
#         return AtA

#     A, b = LHS[()], RHS

    A = eta_T_H_stacked
    b = b_ls
    Atb = A.T.dot(b_ls)

    x0 = u_cur#np.zeros(AtA.shape[1])

    bnds = [(0, np.inf)]*x0.shape[0]

    
    res = scipy.optimize.minimize(fun, x0, args=(A, b, AA, Atb, S, StS, lambda_smoothing), tol = 1e-5, method='L-BFGS-B', jac=grad, bounds=bnds,
               options = {'maxiter': nnls_max_iter, iprint: 0})#'disp':True})
    print(res)

    end = time.time()
    print('u update took:', end - start)
    u_next = res.x
    return u_next

#update in w0:
def w_0_update(eta_0, u, T, alpha, B):
    """Computes the update for the auxilary tumor dose distribution w_0 based on the current u

    Parameters
    ----------
    eta_0 : float
        Penalty parameter for the tumor
    u : np.array of shape (None,)
        Beamlet radiation distribution u
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix block_diag(T_1, ... , T_M) 
        where T_i is the sparse dose deposition matrix for i-th modality
    alpha : np.array of shape (None,)
        Linear block coefficient array of the tumor BE 
        (with N_m included as coefficients)
    B : np.array of shape (None,)
        Quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m included as coefficients)
   
    Returns
    -------
    np.array of shape (None,)
        update for the auxilary tumor dose distribution w_0

    
    """
    return prox(eta_0, T.dot(u) - eta_0*alpha, B)

# update in w:
def w_update(u, H, gamma, D, C):
    """Computes the updates for generalized OAR auxilary dose distributions

    Parameters
    ----------
    u : np.array of shape (None,)
        Beamlet radiation distribution u
    H : List of sparse arrays of shape(None, None)
        List of block-diagonal generalized OAR dose-deposition matrices block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality for a given OAR
    gamma : List of np.arrays of shape (None,)
        List of linear block coefficient array of the generalized OAR BE 
        (with N_m included as coefficients)
    D : List of np.arrays of shape (None,)
        List of quadratic block-diagonal coefficient arrays D of the generalized OAR BEs
        (with N_m included as coefficients)
    C : array-like of floats
        Constraints to satisfy
   
    Returns
    -------
    w_next : list of np.array of shape (None,)
        updates for generalized OAR auxilary dose distributions

    
    """
    import time

    start = time.time()
    
    w_next = [proj(H[i].dot(u), gamma[i], D[i], C[i]) for i in range(len(H))]
    
    end = time.time()
    print('wi update took:', end - start)
    return w_next

# #Parallel update in w:
# import multiprocessing
# from joblib import Parallel, delayed
# from tqdm import tqdm


# def w_update(u, H, gamma, D, C):
#     """Computes the updates for generalized OAR auxilary dose distributions

#     Parameters
#     ----------
#     u : np.array of shape (None,)
#         Beamlet radiation distribution u
#     H : List of sparse arrays of shape(None, None)
#         List of block-diagonal generalized OAR dose-deposition matrices block_diag(H_1, ... , H_M) 
#         where H_i is the sparse dose deposition matrix for i-th modality for a given OAR
#     gamma : List of np.arrays of shape (None,)
#         List of linear block coefficient array of the generalized OAR BE 
#         (with N_m included as coefficients)
#     D : List of np.arrays of shape (None,)
#         List of quadratic block-diagonal coefficient arrays D of the generalized OAR BEs
#         (with N_m included as coefficients)
#     C : array-like of floats
#         Constraints to satisfy
   
#     Returns
#     -------
#     w_next : list of np.array of shape (None,)
#         updates for generalized OAR auxilary dose distributions

    
#     """
#     num_cores = multiprocessing.cpu_count()
#     inputs = tqdm(range(len(H)))
#     def w_plus(i, u, H, gamma, D, C):
#         return proj(H[i].dot(u), gamma[i], D[i], C[i])
#     w_next = Parallel(n_jobs=num_cores)(delayed(w_plus)(i, u, H, gamma, D, C) for i in inputs)
#     return w_next

    
#Update for w_smoothing
def w_s_update(u, S):
    """Computes the updates for the proxy of the derivative of u for smoothing

    Parameters
    ----------
    u : np.array of shape (None,)
        Beamlet radiation distribution u
    S : np.array of shape (None, None)   
    Returns
    -------
    w_next : list of np.array of shape (None,)
        updates for generalized OAR auxilary dose distributions

    
    """
    w_next = S.dot(u)
    w_next[w_next>0] = 0
    return w_next





#SOLVER:

# The following is an attempt to incorporate smoothing matrix
# #for given N1, N2, eta0, eta
# def solver(u_init, eta_s, eta_0, eta, S, T, H, alpha, gamma, B, D, C, ftol = 1e-3, max_iter = 5000, verbose = 0):
#     """Returns the optimal u for the relaxed problem in section 3.2.1 of the paper

#     Parameters
#     ----------
#     u_init : np.array of shape (None,)
#         Initial guess for the beamlet radiation distribution u
#     eta_0 : float
#         Penalty parameter for the tumor
#     eta : np.array of shape (None,)
#         Array of OAR penalty parameters
#     T : Sparse array of shape(None, None)
#         Block-diagonal tumor dose-deposition matrix block_diag(T_1, ... , T_M) 
#         where T_i is the sparse dose deposition matrix for i-th modality
#     H : List of sparse arrays of shape(None, None)
#         List of block-diagonal generalized OAR dose-deposition matrices block_diag(H_1, ... , H_M) 
#         where H_i is the sparse dose deposition matrix for i-th modality for a given OAR
#     alpha : np.array of shape (None,)
#         Linear block coefficient array of the tumor BE 
#         (with N_m included as coefficients)
#     gamma : List of np.arrays of shape (None,)
#         List of linear block coefficient array of the generalized OAR BE 
#         (with N_m included as coefficients)
#     B : np.array of shape (None,)
#         Quadratic block-diagonal coefficient array B of the tumor BE
#         (with N_m included as coefficients)
#     D : List of np.arrays of shape (None,)
#         List of quadratic block-diagonal coefficient arrays D of the generalized OAR BEs
#         (with N_m included as coefficients)
#     C : array-like of floats
#         Constraints to satisfy
#     ftol : float
#         Relative reduction in the objective value for the stopping criterium
#     max_iter : int
#         Max number of iterations
#     verbose : 0 or 1
#         0 for no printing
#         1 for printing
   
#     Returns
#     -------
#     u : np.array of shape (None,)
#         The optimal u for the relaxed problem in section 3.2.1 of the paper
    
#     """
#     #precompute the expensive operation:
#     eta_S_T_H_stacked = scipy.sparse.vstack([S.multiply(1/np.sqrt(2*eta_s))] + [T.multiply(1/np.sqrt(2*eta_0))] + [H[i].multiply(1/np.sqrt(2*eta[i])) for i in range(len(H))])
#     u_prev = u_init + 1
#     u = u_init
#     count = 0
#     obj_history = []
#     relaxed_obj_history = [-1, 0.1] #just two initial values to enter the loop
#     while np.abs((relaxed_obj_history[-2] - relaxed_obj_history[-1])/relaxed_obj_history[-2]) > ftol and count < max_iter:
#         start = time.time()
        
#         u_prev = np.copy(u)
#         w_s = w_s_update(u, S)
#         w_0 = w_0_update(eta_0, u, T, alpha, B) 
#         w = w_update(u, H, gamma, D, C) 

#         u = u_update(eta_s, eta_0, eta, w_s, w_0, w, eta_S_T_H_stacked, nnls_max_iter=30)

#         # u = u_update(eta_0, eta, w_s, w_0, w, eta_S_T_H_stacked, nnls_max_iter=30)

#         count += 1 
#         if count == 10:
#             u_inf = np.copy(u)
#         if count > 10 and np.abs(cur_obj) > 1e+15:
#             print('INFINITY! RETURNING u at the 10-th iteration to enter the feasibility loop')
#             return u_inf, obj_history, relaxed_obj_history
        
#         cur_obj = obj_u_opt_N_fixed(u, T, alpha, B)
#         obj_history.append(cur_obj)
#         cur_relaxed_obj = relaxed_obj_u_opt_N_fixed(u, w_0, w, eta_0, eta, T, H, alpha, B)
#         relaxed_obj_history.append(cur_relaxed_obj)    
        
#         stop = time.time()
#         duration = stop-start
        
#         if count%1 == 0 and verbose: 
#             stopping_criterium = np.abs((relaxed_obj_history[-2] - relaxed_obj_history[-1])/relaxed_obj_history[-2])
#             print('    iter = {}, stopping criterium:{}, OBJ {}'.format(count, stopping_criterium, cur_obj))
#             print('    This iteration took: {}'.format(duration))
#     return u, obj_history, relaxed_obj_history


def solver(u_init, S, StS, lambda_smoothing, eta_0, eta, T, H, alpha, gamma, B, D, C, ftol = 1e-3, max_iter = 5000, verbose = 0):
    """Returns the optimal u for the relaxed problem in section 3.2.1 of the paper

    Parameters
    ----------
    u_init : np.array of shape (None,)
        Initial guess for the beamlet radiation distribution u
    eta_0 : float
        Penalty parameter for the tumor
    eta : np.array of shape (None,)
        Array of OAR penalty parameters
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix block_diag(T_1, ... , T_M) 
        where T_i is the sparse dose deposition matrix for i-th modality
    H : List of sparse arrays of shape(None, None)
        List of block-diagonal generalized OAR dose-deposition matrices block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality for a given OAR
    alpha : np.array of shape (None,)
        Linear block coefficient array of the tumor BE 
        (with N_m included as coefficients)
    gamma : List of np.arrays of shape (None,)
        List of linear block coefficient array of the generalized OAR BE 
        (with N_m included as coefficients)
    B : np.array of shape (None,)
        Quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m included as coefficients)
    D : List of np.arrays of shape (None,)
        List of quadratic block-diagonal coefficient arrays D of the generalized OAR BEs
        (with N_m included as coefficients)
    C : array-like of floats
        Constraints to satisfy
    ftol : float
        Relative reduction in the objective value for the stopping criterium
    max_iter : int
        Max number of iterations
    verbose : 0 or 1
        0 for no printing
        1 for printing
   
    Returns
    -------
    u : np.array of shape (None,)
        The optimal u for the relaxed problem in section 3.2.1 of the paper
    
    """
    #precompute the expensive operation:
    eta_T_H_stacked = scipy.sparse.vstack([T.multiply(1/np.sqrt(2*eta_0))] + [H[i].multiply(1/np.sqrt(2*eta[i])) for i in range(len(H))])
    AtA = eta_T_H_stacked.T.dot(eta_T_H_stacked)
    AA = AtA.toarray()
    #!!!!
#     premultiplied_lhs = eta_T_H_stacked.T.dot(eta_T_H_stacked).toarray()
    #!!!!
    u_prev = u_init + 1
    u = u_init
    count = 0
    obj_history = []
    relaxed_obj_history = [-1, 0.1] #just two initial values to enter the loop
    while np.abs((relaxed_obj_history[-2] - relaxed_obj_history[-1])/relaxed_obj_history[-2]) > ftol and count < max_iter:#np.linalg.norm(u - u_prev, np.inf) > 1e-3 and count < max_iter: #Maybe all of them stop changing
        start = time.time()
        
        u_prev = np.copy(u)
        w_0 = w_0_update(eta_0, u, T, alpha, B) 
        w = w_update(u, H, gamma, D, C)
        nnls_max_iter = 50 + (max(0,count-10))**(3/2)
        u = u_update(u, AtA, AA, S, StS, lambda_smoothing, eta_0, eta, w_0, w, eta_T_H_stacked, nnls_max_iter=nnls_max_iter)
#         u = u_update(eta_0, eta, w_0, w, eta_T_H_stacked, nnls_max_iter=50)
        #!!!!
#         u = u_update(eta_0, eta, w_0, w, eta_T_H_stacked, nnls_max_iter=30)
        #!!!!
        count += 1 
        if count == 10:
            u_inf = np.copy(u)
        if count > 10 and np.abs(cur_obj) > 1e+15: #HANDLE THIS BETTER!!!
            print('INFINITY! RETURNING u at the 10-th iteration to enter the feasibility loop')
            return u_inf, obj_history, relaxed_obj_history
        
        cur_obj = obj_u_opt_N_fixed(u, T, alpha, B)
        obj_history.append(cur_obj)
        cur_relaxed_obj = relaxed_obj_u_opt_N_fixed(u, w_0, w, eta_0, eta, T, H, alpha, B)
        relaxed_obj_history.append(cur_relaxed_obj)    
        
        stop = time.time()
        duration = stop-start
        
        if count%1 == 0 and verbose: 
            stopping_criterium = np.abs((relaxed_obj_history[-2] - relaxed_obj_history[-1])/relaxed_obj_history[-2])
            print('    iter = {}, stopping criterion:{}, OBJ {}'.format(count, stopping_criterium, cur_obj))
            print('    This iteration took: {}'.format(duration))
    return u, obj_history, relaxed_obj_history

def check_photon_target_smoothness(target_photon_matrix, u_mult_smoothed, max_min_ratio = 2, proton_only = False):
    """max_min_ratio -- the amount of difference we allow between max and min tumor dose"""
    if proton_only:
        True
    target_dose = target_photon_matrix.dot(u_mult_smoothed[:target_photon_matrix.shape[1]])
    print('SMOOTHING, max/min ratio in the target dose:', np.max(target_dose)/np.min(target_dose))
    if np.max(target_dose)/np.min(target_dose) <= max_min_ratio:
        return True
    else:
        return False

    
#Automatic choice of etas:

def solver_auto_param(u_init, target_photon_matrix, S, StS, lambda_smoothing, smoothing_ratio, T, H, alpha, gamma, B, D, C, eta_step = 0.5, ftol = 1e-3, max_iter = 300, eta_0 = None, eta = None, verbose = 0, proton_only = False):
    """Returns the optimal u for the relaxed problem in section 3.2.1 of the paper
    with the automated parameter selection

    Parameters
    ----------
    u_init : np.array of shape (None,)
        Initial guess for the beamlet radiation distribution u
    T : Sparse array of shape(None, None)
        Block-diagonal tumor dose-deposition matrix block_diag(T_1, ... , T_M) 
        where T_i is the sparse dose deposition matrix for i-th modality
    H : List of sparse arrays of shape(None, None)
        List of block-diagonal generalized OAR dose-deposition matrices block_diag(H_1, ... , H_M) 
        where H_i is the sparse dose deposition matrix for i-th modality for a given OAR
    alpha : np.array of shape (None,)
        Linear block coefficient array of the tumor BE 
        (with N_m included as coefficients)
    gamma : List of np.arrays of shape (None,)
        List of linear block coefficient array of the generalized OAR BE 
        (with N_m included as coefficients)
    B : np.array of shape (None,)
        Quadratic block-diagonal coefficient array B of the tumor BE
        (with N_m included as coefficients)
    D : List of np.arrays of shape (None,)
        List of quadratic block-diagonal coefficient arrays D of the generalized OAR BEs
        (with N_m included as coefficients)
    C : array-like of floats
        Constraints to satisfy
    eta_step : float
        Reduction rate for the penalty parameters
    ftol : float
        Relative reduction in the objective value for the stopping criterium in the inner fixed-parameter solver
    max_iter : int
        Max number of iterations for the inner fixed-parameter solver
    verbose : 0 or 1
        0 for no printing
        1 for printing
    proton_only : bool
        if True, the photon target smoothness is never enforced, False by default
   
    Returns
    -------
    u : np.array of shape (None,)
        The optimal u for the relaxed problem in section 3.2.1 of the paper
    eta_0 : float
        The optimal tumor penalty parameter
    eta : np.array of shape (None, )
        The optimal generalized OAR penalty parameter
    auto_param_obj_history : list of lists 
        Objective histories
    auto_param_relaxed_obj_history : list of lists
        Relaxed objective histories
    """
    auto_param_obj_history = []
    auto_param_relaxed_obj_history = []
    if eta_0 is None:
        eta_0 =  (1/(2*np.max(B)))*0.5 #Initialize eta_0
    if eta is None:
        eta = np.array([eta_0/len(H)]*len(H))*2#*0.01#np.array([eta_0/len(H)]*len(H))*2 
    
    u, obj_history, relaxed_obj_history = solver(u_init, S, StS, lambda_smoothing, eta_0, eta, T, H, alpha, gamma, B, D, C, ftol = 1e-3, max_iter = max_iter, verbose = verbose)
    auto_param_obj_history.append(obj_history)
    auto_param_relaxed_obj_history.append(relaxed_obj_history)
    cnstr = constraints_all(u, H, gamma, D, C, tol = 0.05, verbose = 0)
    photon_target_smoothness = check_photon_target_smoothness(target_photon_matrix, u, max_min_ratio = smoothing_ratio, proton_only = proton_only)
    
    print('Enforcing Feasibility')
    count = 0
    num_violated = -1
    while (cnstr['Relaxed'].sum()-len(H)) or (not photon_target_smoothness):
        count += 1
        num_violated_prev = np.copy(num_violated)
        num_violated = cnstr['Relaxed'].sum() - len(H)
        
        print('Iter ', count, '# of violated constr:', cnstr['Relaxed'].sum()-len(H))
        eta[cnstr['Relaxed'] == False] *= eta_step
        
        if not photon_target_smoothness:
            lambda_smoothing *= (1/eta_step)
            
        if num_violated == num_violated_prev:
            print('Increase enforcement')
            eta[cnstr['Relaxed'] == False] *= eta_step
            lambda_smoothing *= (1/eta_step)
            print('Lambda Smoothing:', lambda_smoothing)
            
        u, obj_history, relaxed_obj_history = solver(u, S, StS, lambda_smoothing, eta_0, eta, T, H, alpha, gamma, B, D, C, ftol = ftol, max_iter = max_iter, verbose = verbose)
        
        auto_param_obj_history.append(obj_history)
        auto_param_relaxed_obj_history.append(relaxed_obj_history)
        
        cnstr = constraints_all(u, H, gamma, D, C, tol = 0.05, verbose = 0)
        photon_target_smoothness = check_photon_target_smoothness(target_photon_matrix, u, max_min_ratio = smoothing_ratio, proton_only = proton_only)
        
    print('Enforcing Optimality')
    count = 0
    while not (cnstr['Relaxed'].sum()-len(H)): #If nothing is violated -- enforce optimality!
        count += 1
        print('Opt Iter', count)
        obj_prev = obj_u_opt_N_fixed(u, T, alpha, B)
        u_prev = np.copy(u)
        eta_0 *= eta_step
        if not photon_target_smoothness:
            print('Smoothing Warning: Smoothing violated, increasing penalty')
            lambda_smoothing *= (1/eta_step)
        
        #Could do a while loop for smoothing here since we have the objective check later anyway
        u, obj_history, relaxed_obj_history = solver(u, S, StS, lambda_smoothing, eta_0, eta, T, H, alpha, gamma, B, D, C, ftol = ftol, max_iter = max_iter, verbose = verbose)
        auto_param_obj_history.append(obj_history)
        #Note that smoothing is currently not counted in the relaxed objective
        auto_param_relaxed_obj_history.append(relaxed_obj_history)
        
        obj_new = obj_u_opt_N_fixed(u, T, alpha, B)
        if (abs(obj_new - obj_prev)/abs(obj_prev) <= 1e-4) or (obj_new > obj_prev): #two consequent iters, two times bc on iter 2 it stops anyway
            print('No improvement, exiting')
            break
            
        cnstr = constraints_all(u, H, gamma, D, C, tol = 0.05, verbose = 0)
        photon_target_smoothness = check_photon_target_smoothness(target_photon_matrix, u, max_min_ratio = smoothing_ratio, proton_only = proton_only)
        print('Resulting lambda_smoothing:', lambda_smoothing)
        print('# of violated constr:', cnstr['Relaxed'].sum()-len(H))
        
    print('Finding the correct solution:')
    u = u_prev
    eta_0 = eta_0/eta_step
    
    cnstr = constraints_all(u, H, gamma, D, C, tol = 0.05, verbose = 0)
    print('# of violated constr:', cnstr['Relaxed'].sum()-len(H))
    print("OBJJJJJ:", obj_u_opt_N_fixed(u, T, alpha, B))
    return u, eta_0, eta, auto_param_obj_history, auto_param_relaxed_obj_history

