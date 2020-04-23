from scipy.optimize import SR1, BFGS
import copy
import scipy.optimize
from optimization_tools import *
import argparse



def construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data, modality_names):
    """Construct input coefficients to the auto-paramsolver based on the OAR constraints, radiation properties and
    current vector of fractions N. Takes as an input information about the tumor and OARs and types of constraints.
    Returns the dose-deposition matrices as well as linear and quadratic coefficeints for every generalized OAR.
    
    Parameters
    ----------
    N : np.array of shape (M,), where M is the number of modalities
        Current vector of fractions, if zeroes are encountered, they will be thrown away
    Alpha : np.array of shape (M,), where M is the number of modalities
        Linear tumor-deposition coefficients for each modality
    Beta : np.array of shape (M,), where M is the number of modalities
        Quadratic tumor-deposition coefficients for each modality
    Gamma : List of np.arrays of shape (M,), where M is the number of modalities
        Linear OAR-deposition coefficients for each modality
    Delta : List of np.arrays of shape (M,), where M is the number of modalities
        Quadratic OAR-deposition coefficients for each modality
    data : dictionary
        Dictionary of the matlab input data from MatRad (citation here!) with 
        example keys: 'Aproton', 'Aphoton', 'Organ', 'num_beamlets', 'num_voxels', 
        'OAR_constraint_types', 'OAR_constraint_values'
    modality_names : list
        List of modality names to extract the dose deposition matrices from the data
        
    Returns
    -------
    T_list : list of sparse arrays of shape (None, None)
        List of tumor-deposition matrices
    T : sparse array of shape (None, None)
        Block-diagonal tumor deposition matrix
    H : list of sparse arrays of shape (None, None)
        List of block-diagonal OAR-deposition matrices
    alpha : array of shape (None,)
        Block-coefficient array of linear tumor-deposition coefficients 
        (with N included as coefficients!)
    gamma : List of arrays of shape (None,)
        List of block-coefficient arrays of linear generalized OAR-deposition coefficients 
        (with N included as coefficients!)
    B : array of shape (None,)
        Block-coefficient array of quadratic tumor-deposition coefficients 
        (with N included as coefficients!)
    D : List of arrays of shape (None,)
        List of block-coefficient arrays of quadratic generalized OAR-deposition coefficients 
        (with N included as coefficients!)
    C : array of shape (None,)
        Constraints array for the generalized OARs  
        
    Notes
    --------
        TODO: possibly add a way to handle different tumor voxel number for different modalities
        NOTE: Could make this independent of N to not reconstruct every time, probably won't save much time

    """
    #First, choose only active modalities, that is, for N!=0, throw away zeros
    active_modalities = np.array(N[:]) > 0
    Alpha = Alpha[active_modalities]
    Beta = Beta[active_modalities]
    Gamma = Gamma[:,active_modalities]
    Delta = Delta[:,active_modalities]
    modality_names = modality_names[active_modalities]
    N = N[active_modalities]
    
    #Construct the tumor dose deposition matrices
    #assuming num_voxels[0] is ALWAYS the tumor!
    num_voxels = np.squeeze(data['num_voxels'])
    T_list = [scipy.sparse.csr_matrix(data[modality][:num_voxels[0]]) for modality in modality_names]

    
    #Tumor linear coeffs:
    #Here we could potentially have different number of voxels, 
    #assume for now the tumor voxel number is the same for each modality
    #We also include -N as coeffs and are consistent with alpha_tilde from the paper with -N_m in coeffs
    alpha = np.repeat(Alpha,  num_voxels[0]) * np.repeat(-N, num_voxels[0]) 
    #No minus in tumor quadratic coeffs B:
    B = np.repeat(Beta,  num_voxels[0]) * np.repeat(N, num_voxels[0])
    
    #Now, we need to construct our generalized OARs
    #OAR linear coeffs (again, assuming, the number of voxels is the same for each modality):
    #Note here that num_voxels[1:] are the OAR voxel numbers
    #(If mean dose: quadr_OAR = lin_OAR/5, if max dose: quadr_OAR = lin_OAR/2)
    gamma = []
    D = []
    C = []
    generalized_num_voxels = [num_voxels[0]]
    OAR_constr_types = np.squeeze(data['OAR_constraint_types'])
    OAR_constr_values = np.squeeze(data['OAR_constraint_values'])
    def BE_constr(d, lin, quad):
        return lin*d + quad*d**2
    for constr_type in range(len(OAR_constr_types)):
        if OAR_constr_types[constr_type].strip() == 'max_dose':
            for voxel in range(num_voxels[1:][constr_type]): #for the max_dose, treat every voxel as a gen_OAR
                #N_m included in coeffs
                gamma.append(np.repeat(Gamma[constr_type], 1) * np.repeat(N, 1))
                D.append(np.repeat(Delta[constr_type], 1) * np.repeat(N, 1))
                #Add the BE constraint for OARs
                total_N = 45# 45 fractions of Photons is the default #np.sum(N)
                d = np.squeeze(OAR_constr_values[constr_type])/total_N
                lin = Gamma[constr_type][0]*total_N
                quad = Delta[constr_type][0]*total_N                
                constr = BE_constr(d, lin, quad)
                C.append(constr)
                generalized_num_voxels.append(1) #1-voxel OAR
        if OAR_constr_types[constr_type].strip() == 'mean_dose':
            gamma.append(np.repeat(Gamma[constr_type], num_voxels[1:][constr_type]) 
                         * np.repeat(N, num_voxels[1:][constr_type]))
            D.append(np.repeat(Delta[constr_type], num_voxels[1:][constr_type]) 
                         * np.repeat(N, num_voxels[1:][constr_type]))
            
            total_N = 45 # 45 fractions of Photons is the default #np.sum(N)
            d = np.squeeze(OAR_constr_values[constr_type])/total_N
            lin = Gamma[constr_type][0]*total_N
            quad = Delta[constr_type][0]*total_N                
            constr = BE_constr(d, lin, quad)
            C.append(constr*num_voxels[1:][constr_type])
#             C.append(OAR_constr_values[constr_type]*num_voxels[1:][constr_type])
            generalized_num_voxels.append(num_voxels[1:][constr_type])
            
    #Construct the dose-deposition matrices for our generalized OARs!
    #The code assumes that every OAR has max or mean dose constraint!
    cur_OAR_index = num_voxels[0]
    OAR_indeces = []
    print('getting through {} matrices'.format(len(generalized_num_voxels)))
    for i in range(len(generalized_num_voxels)-1):
        cur_OAR_index += generalized_num_voxels[i]
        left = cur_OAR_index 
        right = cur_OAR_index+generalized_num_voxels[i+1]
        OAR_indeces.append((left,right))
    H_list = [[scipy.sparse.csr_matrix(data[modality][OAR[0]:OAR[1]]) for OAR in OAR_indeces] 
          for modality in modality_names]
    #The dose deposition matrices:
    T = scipy.sparse.block_diag(T_list)
    H = [scipy.sparse.block_diag(Hi) for Hi in zip(*H_list)]
    return T_list, T, H, alpha, gamma, B, D, C
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default = 0.35, type=float)
    parser.add_argument('--beta', default = 0.175, type=float)
    parser.add_argument('--gamma', default = 0.35, type=float)
    parser.add_argument('--delta_mean', default = 0.07, type=float)
    parser.add_argument('--delta_max', default = 0.175, type=float)
    parser.add_argument('--data_path', default = 'data/ProstateExample_BODY_not_reduced_with_OAR_constraints.mat')

    args = parser.parse_args()
    
    data = scipy.io.loadmat(args.data_path)


    Alpha = np.array([args.alpha, args.alpha])
    Beta = np.array([args.beta, args.beta])
    Gamma = np.array([np.array([args.gamma, args.gamma]),
                      np.array([args.gamma, args.gamma]),
                      np.array([args.gamma, args.gamma]),
                      np.array([args.gamma, args.gamma]),
                      np.array([args.gamma, args.gamma])               
                     ])
    Delta = np.array([np.array([args.delta_mean, args.delta_mean]),
                      np.array([args.delta_mean, args.delta_mean]),
                      np.array([args.delta_max, args.delta_max]),
                      np.array([args.delta_max, args.delta_max]),
                      np.array([args.delta_max, args.delta_max])                
                     ])
    modality_names = np.array(['Aphoton', 'Aproton'])


    def objective_N(N,  Alpha = Alpha, Beta = Beta, Gamma = Gamma, Delta = Delta, data = data, modality_names = modality_names):
        """Value function of N as per section 3.1 of the paper

        Parameters
        ----------
        N : np.array of shape (M,), where M is the number of modalities
            Current vector of fractions, if zeroes are encountered, they will be thrown away
        Alpha : np.array of shape (M,), where M is the number of modalities
            Linear tumor-deposition coefficients for each modality
        Beta : np.array of shape (M,), where M is the number of modalities
            Quadratic tumor-deposition coefficients for each modality
        Gamma : List of np.arrays of shape (M,), where M is the number of modalities
            Linear OAR-deposition coefficients for each modality
        Delta : List of np.arrays of shape (M,), where M is the number of modalities
            Quadratic OAR-deposition coefficients for each modality
        data : dictionary
            Dictionary of the matlab input data from MatRad (citation here!) with 
            example keys: 'Aproton', 'Aphoton', 'Organ', 'num_beamlets', 'num_voxels', 
            'OAR_constraint_types', 'OAR_constraint_values'
        modality_names : list
            List of modality names to extract the dose deposition matrices from the data

        Returns
        -------
        float
            value function V (from the section 3.1 of the paper) evaluated at N
        """
        T_list, T, H, alpha, gamma, B, D, C = construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data, modality_names)
        num_tumor_voxels = np.squeeze(data['num_voxels'])[0]
        #Assuming N>0, and two-modality case
        Rx = 80
        LHS1 = T_list[0]
        LHS2 = T_list[1]
        RHS1 = np.array([Rx/(np.sum(N))]*LHS1.shape[0])
        RHS2 = np.array([Rx/(np.sum(N))]*LHS2.shape[0])


        u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-2, max_iter=30, verbose=1).x
        u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-2, max_iter=30, verbose=1).x


        u_init11 = np.concatenate([u1_guess, u2_guess])


        u, eta_0, eta, auto_param_obj_history, auto_param_relaxed_obj_history = solver_auto_param(u_init11, T, H, alpha, gamma, B, D, C, eta_step = 0.1, ftol = 1e-3, max_iter = 300, verbose = 1)

        return obj_u_opt_N_opt(u, T, alpha, B, N, num_tumor_voxels, Td = 10)


        

    def optimize_N(N_init, N_SUM_MAX = 45, max_iter = 6):
        """Value function of N optimization as per Algorithm 1 of the paper.
        Uses trust-region-constrained from scipy

        Parameters
        ----------
        N_init : np.array of shape (M,), where M is the number of modalities
            Initial array of fractions
        N_SUM_MAX : float
            Max treatment course length, the bound on the sum of N
        max_iter : int
            Max number of iterations

        Returns
        -------
        x : np.array of shape (M,)
            Optimal fractionation schedule N
        res : OptimizationResult from scipy
            Metadata of the algorithm
        Nk_hist : list
            Iterates N history 
        """
        Nk_hist = []
        def callback(xk, OptRes):
            Nk_hist.append(xk)
            return 0

        bounds =  scipy.optimize.Bounds([1, 1], [np.inf, np.inf]) #or 0,0
        linear_constraint = scipy.optimize.LinearConstraint(np.array([1,1]), -np.inf, N_SUM_MAX)

        x0 = np.array(N_init)
        res = scipy.optimize.minimize(objective_N, x0, method='trust-constr',  jac="2-point", hess=BFGS(), 
                             constraints=[linear_constraint], options={'verbose': 4, 'maxiter': max_iter}, bounds=bounds, 
                             callback = callback)
        x = res.x
        print('NEW N rounded:', np.rint(x), 'EXACT new N:', x)
        return x, res, Nk_hist

    x, res, Nk_hist = optimize_N(np.array([44,1]), N_SUM_MAX = 45, max_iter = 6)
