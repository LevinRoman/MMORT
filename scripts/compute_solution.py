"""Script to compute treatment plan
ARGS: 
	data_path
	coefficients_path (for Alpha, Beta, Gamma, Delta)
	N1
	N2
	Rx_init
	lambda_smoothing
	"""

import argparse
import numpy as np
import copy
import scipy.optimize
import pandas as pd
import operator
import scipy.io
import scipy
import scipy.sparse
import time
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
#Import mmort modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'mmort')))
import experiments
import optimization_tools
import evaluation
import utils
from config import configurations


if __name__ == '__main__':
	print(np.show_config())
	parser = argparse.ArgumentParser()



	parser.add_argument('--data_name', default = 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat', type = str)
	parser.add_argument('--config_experiment', default = 'Experiment_1', type = str)
	parser.add_argument('--smoothing_ratio', default = 2.0, type = float)
	parser.add_argument('--lambda_smoothing', default = 1e5, type = float)
	parser.add_argument('--precomputed_input', default = 'no', type = str)
	parser.add_argument('--precomputed_solution', default = 'no', type = str)
	parser.add_argument('--N1', default = 43.0, type = float)
	parser.add_argument('--N2', default = 1.0, type = float)
	parser.add_argument('--N_photon', default = 44.0, type = float)
	parser.add_argument('--N_proton', default = 44.0, type = float)
	parser.add_argument('--compute_proton', default = 'yes', type = str)
	parser.add_argument('--compute_photon', default = 'yes', type = str)
	parser.add_argument('--compute_proton', default = 'yes', type = str)
	parser.add_argument('--Rx', default = 190.0, type = float)
	parser.add_argument('--eta0_coef_proton', default = 0.9, type = float)
	parser.add_argument('--eta_coef_proton', default = 1e-7, type = float)
	parser.add_argument('--eta0_coef_photon', default = 0.9, type = float)
	parser.add_argument('--eta_coef_photon', default = 1e-7, type = float)
	parser.add_argument('--eta0_coef_proton', default = 0.9, type = float)
	parser.add_argument('--eta_coef_proton', default = 1e-7, type = float)
	parser.add_argument('--eta_step', default = 0.1, type = float)
	parser.add_argument('--ftol', default = 1e-3, type = float)
	parser.add_argument('--max_iter', default = 50.0, type = float)
	parser.add_argument('--update_parameters', default = 'no', type = str)

 	# eta_step = 0.1, ftol = 1e-3, max_iter = 50, verbose = 1


	args = parser.parse_args()


	data_name = args.data_name
	config_experiment = args.config_experiment
	smoothing_ratio = args.smoothing_ratio
	lambda_smoothing = args.lambda_smoothing
	precomputed_input = args.precomputed_input
	precomputed_solution = args.precomputed_solution
	N1 = args.N1
	N2 = args.N2
	N_photon = args.N_photon
	N_proton = args.N_proton
	N = np.array([N1, N2])
	compute_proton = args.compute_proton
	compute_photon = args.compute_photon
	compute_proton = args.compute_proton
	Rx = args.Rx
	eta0_coef_proton = args.eta0_coef_proton
	eta_coef_proton = args.eta_coef_proton
	eta0_coef_photon = args.eta0_coef_photon
	eta_coef_photon = args.eta_coef_photon
	eta0_coef_proton = args.eta0_coef_proton
	eta_coef_proton = args.eta_coef_proton
	eta_step = args.eta_step
	ftol = args.ftol
	max_iter = args.max_iter
	update_parameters = args.update_parameters


	print('Args: data_name={}, config_experiment={}, smoothing_ratio={}, precomputed_input={}, N1={}, N2={}, N_photon={}, N_proton={}'.format(data_name, config_experiment, smoothing_ratio, precomputed_input, N1, N2, N_photon, N_proton))


	###########################
	#Raw Data Part
	###########################

	#Load data
	data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', data_name))
	data = scipy.io.loadmat(data_path)
	
	#Adjust the last row of the dose matrices to make sure the BODY is not the sum row but the mean row
	num_body_voxels = 683189 #It is very bad that this is hard coded, will adjust the data file permanently later
	data['Aphoton'][-1] = data['Aphoton'][-1]/num_body_voxels
	data['Aproton'][-1] = data['Aproton'][-1]/num_body_voxels
	data['OAR_constraint_fraction'] = [0.5, 0.5, 1.0, 1.0, 1.0] #Added dv constraint fraction (1.0 for max-dose) for evaluation

	print('\nData loaded from '+data_path)


	#Create a copy of the data with max dose for dose-volume constrained organs, to faster adjust dv for :
	data_max_dose = copy.deepcopy(data)
	data_max_dose['OAR_constraint_types'][data_max_dose['OAR_constraint_types'] == 'dose_volume'] = 'max_dose'
	

	###########################
	#Experimental Setup Part
	###########################

	#Load experimental setup from config
	experiment_setup = configurations[config_experiment]

	Alpha = experiment_setup['Alpha']
	Beta = experiment_setup['Beta']
	Gamma = experiment_setup['Gamma']
	Delta = experiment_setup['Delta']

	modality_names = experiment_setup['modality_names']

	print('\nExperimental Setup: \nAlpha={} \nBeta={} \nGamma={} \nDelta={} \nModality Names: {}'.format(Alpha, Beta, Gamma, Delta, modality_names))


	############################
	#Form input matrices
	############################


	if compute_proton == 'yes':
		start = time.time()
		#Initial input, with dv constraint types, protoni-modality
		if precomputed_input == 'no':
			T_list_proton, T_proton, H_proton, alpha_proton, gamma_proton, B_proton, D_proton, C_proton = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data, modality_names)
			saving_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			utils.save_obj(T_list_proton, 'T_list_proton', saving_dir)
			utils.save_obj(T_proton, 'T_proton', saving_dir)
			utils.save_obj(H_proton, 'H_proton', saving_dir)
			utils.save_obj(alpha_proton, 'alpha_proton', saving_dir)
			utils.save_obj(gamma_proton, 'gamma_proton', saving_dir)
			utils.save_obj(B_proton, 'B_proton', saving_dir)
			utils.save_obj(D_proton, 'D_proton', saving_dir)
			utils.save_obj(C_proton, 'C_proton', saving_dir)
			print('\nInitial input, with dv constraint types, protoni-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			T_list_proton = utils.load_obj( 'T_list_proton', loading_dir)
			T_proton = utils.load_obj('T_proton', loading_dir)
			H_proton = utils.load_obj('H_proton', loading_dir)
			alpha_proton = utils.load_obj('alpha_proton', loading_dir)
			gamma_proton = utils.load_obj('gamma_proton', loading_dir)
			B_proton = utils.load_obj('B_proton', loading_dir)
			D_proton = utils.load_obj('D_proton', loading_dir)
			C_proton = utils.load_obj('C_proton', loading_dir)
			print('\nInitial input, with dv constraint types, protoni-modality loaded from '+loading_dir)

		
		end = time.time()
		print('Time elapsed:', end - start)


		start = time.time()
		#Max Dose for dv constrained organs input, protoni-modality
		if precomputed_input == 'no':
			T_list_proton_max, T_proton_max, H_proton_max, alpha_proton_max, gamma_proton_max, B_proton_max, D_proton_max, C_proton_max = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
			saving_dir = config_experiment+'_proton_max_{}_{}'.format(N1, N2)
			utils.save_obj(T_list_proton_max, 'T_list_proton_max', saving_dir)
			utils.save_obj(T_proton_max, 'T_proton_max', saving_dir)
			utils.save_obj(H_proton_max, 'H_proton_max', saving_dir)
			utils.save_obj(alpha_proton_max, 'alpha_proton_max', saving_dir)
			utils.save_obj(gamma_proton_max, 'gamma_proton_max', saving_dir)
			utils.save_obj(B_proton_max, 'B_proton_max', saving_dir)
			utils.save_obj(D_proton_max, 'D_proton_max', saving_dir)
			utils.save_obj(C_proton_max, 'C_proton_max', saving_dir)
			print('\nMax Dose input for dv constrained organs input, protoni-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_proton_max_{}_{}'.format(N1, N2)
			T_list_proton_max = utils.load_obj('T_list_proton_max', loading_dir)
			T_proton_max = utils.load_obj('T_proton_max', loading_dir)
			H_proton_max = utils.load_obj('H_proton_max', loading_dir)
			alpha_proton_max = utils.load_obj('alpha_proton_max', loading_dir)
			gamma_proton_max = utils.load_obj('gamma_proton_max', loading_dir)
			B_proton_max = utils.load_obj('B_proton_max', loading_dir)
			D_proton_max = utils.load_obj('D_proton_max', loading_dir)
			C_proton_max = utils.load_obj('C_proton_max', loading_dir)
			print('\nMax Dose input for dv constrained organs input, protoni-modality loaded from '+loading_dir)

		end = time.time()
		print('Time elapsed:', end - start)


	if compute_photon == 'yes':
		start = time.time()
		#Initial input, with dv constraint types, photon-modality
		if precomputed_input == 'no':
			T_list_photon, T_photon, H_photon, alpha_photon, gamma_photon, B_photon, D_photon, C_photon = experiments.construct_auto_param_solver_input(np.array([N_photon,0]), Alpha, Beta, Gamma, Delta, data, modality_names)
			saving_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
			utils.save_obj(T_list_photon, 'T_list_photon', saving_dir)
			utils.save_obj(T_photon, 'T_photon', saving_dir)
			utils.save_obj(H_photon, 'H_photon', saving_dir)
			utils.save_obj(alpha_photon, 'alpha_photon', saving_dir)
			utils.save_obj(gamma_photon, 'gamma_photon', saving_dir)
			utils.save_obj(B_photon, 'B_photon', saving_dir)
			utils.save_obj(D_photon, 'D_photon', saving_dir)
			utils.save_obj(C_photon, 'C_photon', saving_dir)
			print('\nInitial input, with dv constraint types, photon-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
			T_list_photon = utils.load_obj('T_list_photon', loading_dir)
			T_photon = utils.load_obj('T_photon', loading_dir)
			H_photon = utils.load_obj('H_photon', loading_dir)
			alpha_photon = utils.load_obj('alpha_photon', loading_dir)
			gamma_photon = utils.load_obj('gamma_photon', loading_dir)
			B_photon = utils.load_obj('B_photon', loading_dir)
			D_photon = utils.load_obj('D_photon', loading_dir)
			C_photon = utils.load_obj('C_photon', loading_dir)
			print('\nInitial input, with dv constraint types, photon-modality loaded from '+loading_dir)

		end = time.time()
		print('Time elapsed:', end - start)

		start = time.time()
		#Max Dose for dv constrained organs input, photon-modality
		if precomputed_input == 'no':
			T_list_photon_max, T_photon_max, H_photon_max, alpha_photon_max, gamma_photon_max, B_photon_max, D_photon_max, C_photon_max = experiments.construct_auto_param_solver_input(np.array([N_photon,0]), Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
			saving_dir = config_experiment+'_photon_max_{}_{}'.format(N_photon, 0)
			utils.save_obj(T_list_photon_max, 'T_list_photon_max', saving_dir)
			utils.save_obj(T_photon_max, 'T_photon_max', saving_dir)
			utils.save_obj(H_photon_max, 'H_photon_max', saving_dir)
			utils.save_obj(alpha_photon_max, 'alpha_photon_max', saving_dir)
			utils.save_obj(gamma_photon_max, 'gamma_photon_max', saving_dir)
			utils.save_obj(B_photon_max, 'B_photon_max', saving_dir)
			utils.save_obj(D_photon_max, 'D_photon_max', saving_dir)
			utils.save_obj(C_photon_max, 'C_photon_max', saving_dir)
			print('\nMax Dose input for dv constrained organs input, photon-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_photon_max_{}_{}'.format(N_photon, 0)
			T_list_photon_max = utils.load_obj('T_list_photon_max', loading_dir)
			T_photon_max = utils.load_obj('T_photon_max', loading_dir)
			H_photon_max = utils.load_obj('H_photon_max', loading_dir)
			alpha_photon_max = utils.load_obj('alpha_photon_max', loading_dir)
			gamma_photon_max = utils.load_obj('gamma_photon_max', loading_dir)
			B_photon_max = utils.load_obj('B_photon_max', loading_dir)
			D_photon_max = utils.load_obj('D_photon_max', loading_dir)
			C_photon_max = utils.load_obj('C_photon_max', loading_dir)
			print('\nMax Dose input for dv constrained organs input, photon-modality loaded from '+loading_dir)

		end = time.time()
		print('Time elapsed:', end - start)
	

	if compute_proton == 'yes':
		start = time.time()
		#Initial input, with dv constraint types, proton-modality
		if precomputed_input == 'no':
			T_list_proton, T_proton, H_proton, alpha_proton, gamma_proton, B_proton, D_proton, C_proton = experiments.construct_auto_param_solver_input(np.array([0, N_proton]), Alpha, Beta, Gamma, Delta, data, modality_names)
			saving_dir = config_experiment+'_proton_{}_{}'.format(0, N_proton)
			utils.save_obj(T_list_proton, 'T_list_proton', saving_dir)
			utils.save_obj(T_proton, 'T_proton', saving_dir)
			utils.save_obj(H_proton, 'H_proton', saving_dir)
			utils.save_obj(alpha_proton, 'alpha_proton', saving_dir)
			utils.save_obj(gamma_proton, 'gamma_proton', saving_dir)
			utils.save_obj(B_proton, 'B_proton', saving_dir)
			utils.save_obj(D_proton, 'D_proton', saving_dir)
			utils.save_obj(C_proton, 'C_proton', saving_dir)
			print('\nInitial input, with dv constraint types, proton-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_proton_{}_{}'.format(0, N_proton)
			T_list_proton = utils.load_obj('T_list_proton', loading_dir)
			T_proton = utils.load_obj('T_proton', loading_dir)
			H_proton = utils.load_obj('H_proton', loading_dir)
			alpha_proton = utils.load_obj('alpha_proton', loading_dir)
			gamma_proton = utils.load_obj('gamma_proton', loading_dir)
			B_proton = utils.load_obj('B_proton', loading_dir)
			D_proton = utils.load_obj('D_proton', loading_dir)
			C_proton = utils.load_obj('C_proton', loading_dir)
			print('\nInitial input, with dv constraint types, proton-modality loaded from '+loading_dir)


		end = time.time()
		print('Time elapsed:', end - start)

		start = time.time()
		#Max Dose for dv constrained organs input, proton-modality
		if precomputed_input == 'no':
			T_list_proton_max, T_proton_max, H_proton_max, alpha_proton_max, gamma_proton_max, B_proton_max, D_proton_max, C_proton_max = experiments.construct_auto_param_solver_input(np.array([N_proton,0]), Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
			saving_dir = config_experiment+'_proton_max_{}_{}'.format(0, N_proton)
			utils.save_obj(T_list_proton_max, 'T_list_proton_max', saving_dir)
			utils.save_obj(T_proton_max, 'T_proton_max', saving_dir)
			utils.save_obj(H_proton_max, 'H_proton_max', saving_dir)
			utils.save_obj(alpha_proton_max, 'alpha_proton_max', saving_dir)
			utils.save_obj(gamma_proton_max, 'gamma_proton_max', saving_dir)
			utils.save_obj(B_proton_max, 'B_proton_max', saving_dir)
			utils.save_obj(D_proton_max, 'D_proton_max', saving_dir)
			utils.save_obj(C_proton_max, 'C_proton_max', saving_dir)
			print('\nMax Dose input for dv constrained organs input, proton-modality saved to '+saving_dir)
		if precomputed_input == 'yes':
			loading_dir = config_experiment+'_proton_max_{}_{}'.format(0, N_proton)
			T_list_proton_max = utils.load_obj('T_list_proton_max', loading_dir)
			T_proton_max = utils.load_obj('T_proton_max', loading_dir)
			H_proton_max = utils.load_obj('H_proton_max', loading_dir)
			alpha_proton_max = utils.load_obj('alpha_proton_max', loading_dir)
			gamma_proton_max = utils.load_obj('gamma_proton_max', loading_dir)
			B_proton_max = utils.load_obj('B_proton_max', loading_dir)
			D_proton_max = utils.load_obj('D_proton_max', loading_dir)
			C_proton_max = utils.load_obj('C_proton_max', loading_dir)
			print('\nMax Dose input for dv constrained organs input, proton-modality loaded from '+loading_dir)


		end = time.time()
		print('Time elapsed:', end - start)


	#######################################
	#Solution computation, protoni-modality
	#######################################
	if compute_proton == 'yes':
		start = time.time()
		#Compute initial guess, protoni-modality
		# Rx = 190#80#190#190 160 120 80
		LHS1 = T_list_proton[0]
		LHS2 = T_list_proton[1]
		RHS1 = np.array([Rx/np.sum(N)]*LHS1.shape[0])
		RHS2 = np.array([Rx/np.sum(N)]*LHS2.shape[0])

		u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x
		u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x

		u_init11 = np.concatenate([u1_guess, u2_guess])
		# u_init11 = np.concatenate([u_conv, np.zeros(u2_guess.shape[0])])

		#Initalize parameters
		eta_0 =  (1/(2*np.max(B_proton)))*eta0_coef_proton#0.9 #Initialize eta_0
		eta = np.array([eta_0/len(H_proton)]*len(H_proton))*eta_coef_proton#1e-7
		lambda_smoothing_init = np.copy(lambda_smoothing)
		#Set up smoothing matrix
		len_voxels = data['Aphoton'].shape[0]
		beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		S = utils.construct_smoothing_matrix(beams, eps = 5)
		S = S.toarray()
		StS = S.T.dot(S)
		# lambda_smoothing = 1e5#1e7#1e-3 #1e-2

		#Compute the solution:
		
		if precomputed_solution == 'no':
			print('\nComputing the solution')
			#First, compute the solution without dv constraint, protoni-modality
			u_proton_smoothed, eta_0_proton_smoothed, eta_proton_smoothed, lambda_smoothing_proton_smoothed, auto_param_obj_history_proton_smoothed, auto_param_relaxed_obj_history_proton_smoothed = optimization_tools.solver_auto_param(u_init11, 
				utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_proton, H_proton, alpha_proton, gamma_proton, B_proton, D_proton, C_proton, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
			saving_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			utils.save_obj(u_proton_smoothed, 'u_proton_smoothed', saving_dir)
			utils.save_obj(eta_0_proton_smoothed, 'eta_0_proton_smoothed', saving_dir)
			utils.save_obj(eta_proton_smoothed, 'eta_proton_smoothed', saving_dir)
			utils.save_obj(lambda_smoothing_proton_smoothed, 'lambda_smoothing_proton_smoothed', saving_dir)
			utils.save_obj(auto_param_obj_history_proton_smoothed, 'auto_param_obj_history_proton_smoothed', saving_dir)
			utils.save_obj(auto_param_relaxed_obj_history_proton_smoothed, 'auto_param_relaxed_obj_history_proton_smoothed', saving_dir)

		if precomputed_solution == 'yes':
			print('\nLoading the solution')
			loading_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			u_proton_smoothed = utils.load_obj('u_proton_smoothed', loading_dir)
			eta_0_proton_smoothed = utils.load_obj('eta_0_proton_smoothed', loading_dir)
			eta_proton_smoothed = utils.load_obj('eta_proton_smoothed', loading_dir)
			# lambda_smoothing_proton_smoothed = 1e12
			lambda_smoothing_proton_smoothed = utils.load_obj('lambda_smoothing_proton_smoothed', loading_dir)
			#Load lambda smoothing here too
			#Try with all the same parameters and initialize larger smoothing

		end = time.time()
		print('\n Mult without DVC Solution Computed. Time elapsed:', end - start)


		#Now with DVH constraints, protoni-modality
		oar_indices, T_list_proton_dv, T_proton_dv, H_proton_dv, alpha_proton_dv, gamma_proton_dv, B_proton_dv, D_proton_dv, C_proton_dv = utils.generate_dose_volume_input(T_list_proton_max, T_proton_max, H_proton_max, alpha_proton_max, gamma_proton_max, B_proton_max, D_proton_max, C_proton_max, u_proton_smoothed, N, data, Alpha, Beta, Gamma, Delta)

		eta_0 =  (1/(2*np.max(B_proton_dv)))*eta0_coef_proton #Initialize eta_0
		eta = np.array([eta_0/len(H_proton_dv)]*len(H_proton_dv))*eta_coef_proton

		#Update parameters
		if update_parameters == 'yes':
			lambda_smoothing_init = np.copy(lambda_smoothing_proton_smoothed)
			eta_0 = eta_0_proton_smoothed
			eta = utils.update_dose_volume_eta(eta, eta_proton_smoothed, oar_indices, data)
		# lambda_smoothing = 1e5

		u_proton_dv, eta_0_proton_dv, eta_proton_dv, lambda_smoothing_proton_dv, auto_param_obj_history_proton_dv, auto_param_relaxed_obj_history_proton_dv = optimization_tools.solver_auto_param(u_proton_smoothed, 
			utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_proton_dv, H_proton_dv, alpha_proton_dv, gamma_proton_dv, B_proton_dv, D_proton_dv, C_proton_dv, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		saving_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
		utils.save_obj(u_proton_dv, 'u_proton_dv', saving_dir)
		utils.save_obj(eta_0_proton_dv, 'eta_0_proton_dv', saving_dir)
		utils.save_obj(eta_proton_dv, 'eta_proton_dv', saving_dir)
		utils.save_obj(lambda_smoothing_proton_dv, 'lambda_smoothing_proton_dv', saving_dir)
		utils.save_obj(auto_param_obj_history_proton_dv, 'auto_param_obj_history_proton_dv', saving_dir)
		utils.save_obj(auto_param_relaxed_obj_history_proton_dv, 'auto_param_relaxed_obj_history_proton_dv', saving_dir)

		end = time.time()
		print('\n Mult DVC Solution Computed. Time elapsed:', end - start)



	#######################################
	#Solution computation, photon modality
	#######################################
	if compute_photon == 'yes':
		start = time.time()
		#Compute initial guess, protoni-modality
		# Rx = 190#80#190#190 160 120 80
		# LHS1 = T_list_proton[0]
		# LHS2 = T_list_proton[1]
		# RHS1 = np.array([Rx/np.sum(N)]*LHS1.shape[0])
		# RHS2 = np.array([Rx/np.sum(N)]*LHS2.shape[0])

		# u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x
		# u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x

		# u_init11 = np.concatenate([u1_guess, u2_guess])
		# u_init11 = np.concatenate([u_conv, np.zeros(u2_guess.shape[0])])

		#Initalize parameters
		eta_0 =  (1/(2*np.max(B_proton)))*eta0_coef_photon#0.9 #Initialize eta_0
		eta = np.array([eta_0/len(H_proton)]*len(H_proton))*eta_coef_photon#1e-7
		lambda_smoothing_init = np.copy(lambda_smoothing)
		#Set up smoothing matrix
		len_voxels = data['Aphoton'].shape[0]
		beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		S = utils.construct_smoothing_matrix(beams, eps = 5)
		S = S.toarray()
		StS = S.T.dot(S)
		# lambda_smoothing = 1e5#1e7#1e-3 #1e-2

		#Compute the solution:
		
		if precomputed_solution == 'no':
			print('\nComputing the solution')
			#First, compute the solution without dv constraint, photoni-modality
			u_photon_smoothed, eta_0_photon_smoothed, eta_photon_smoothed, lambda_smoothing_photon_smoothed, auto_param_obj_history_photon_smoothed, auto_param_relaxed_obj_history_photon_smoothed = optimization_tools.solver_auto_param(u1_guess, 
				utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_photon, H_photon, alpha_photon, gamma_photon, B_photon, D_photon, C_photon, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
			saving_dir = config_experiment+'_photon_{}_{}'.format(N1, N2)
			utils.save_obj(u_photon_smoothed, 'u_photon_smoothed', saving_dir)
			utils.save_obj(eta_0_photon_smoothed, 'eta_0_photon_smoothed', saving_dir)
			utils.save_obj(eta_photon_smoothed, 'eta_photon_smoothed', saving_dir)
			utils.save_obj(lambda_smoothing_photon_smoothed, 'lambda_smoothing_photon_smoothed', saving_dir)
			utils.save_obj(auto_param_obj_history_photon_smoothed, 'auto_param_obj_history_photon_smoothed', saving_dir)
			utils.save_obj(auto_param_relaxed_obj_history_photon_smoothed, 'auto_param_relaxed_obj_history_photon_smoothed', saving_dir)

		if precomputed_solution == 'yes':
			print('\nLoading the solution')
			loading_dir = config_experiment+'_photon_{}_{}'.format(N1, N2)
			u_photon_smoothed = utils.load_obj('u_photon_smoothed', loading_dir)
			eta_0_photon_smoothed = utils.load_obj('eta_0_photon_smoothed', loading_dir)
			eta_photon_smoothed = utils.load_obj('eta_photon_smoothed', loading_dir)
			# lambda_smoothing_photon_smoothed = 1e12
			lambda_smoothing_photon_smoothed = utils.load_obj('lambda_smoothing_photon_smoothed', loading_dir)
			#Load lambda smoothing here too
			#Try with all the same parameters and initialize larger smoothing

		end = time.time()
		print('\n Photon without DVC Solution Computed. Time elapsed:', end - start)


		#Now with DVH constraints, photon-modality
		oar_indices, T_list_photon_dv, T_photon_dv, H_photon_dv, alpha_photon_dv, gamma_photon_dv, B_photon_dv, D_photon_dv, C_photon_dv = utils.generate_dose_volume_input(T_list_photon_max, T_photon_max, H_photon_max, alpha_photon_max, gamma_photon_max, B_photon_max, D_photon_max, C_photon_max, u_photon_smoothed, N, data, Alpha, Beta, Gamma, Delta)

		eta_0 =  (1/(2*np.max(B_photon_dv)))*eta0_coef_photon #Initialize eta_0
		eta = np.array([eta_0/len(H_photon_dv)]*len(H_photon_dv))*eta_coef_photon

		#Update parameters
		if update_parameters == 'yes':
			lambda_smoothing_init = np.copy(lambda_smoothing_photon_smoothed)
			eta_0 = eta_0_photon_smoothed
			eta = utils.update_dose_volume_eta(eta, eta_photon_smoothed, oar_indices, data)
		# lambda_smoothing = 1e5

		u_photon_dv, eta_0_photon_dv, eta_photon_dv, lambda_smoothing_photon_dv, auto_param_obj_history_photon_dv, auto_param_relaxed_obj_history_photon_dv = optimization_tools.solver_auto_param(u_photon_smoothed, 
			utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_photon_dv, H_photon_dv, alpha_photon_dv, gamma_photon_dv, B_photon_dv, D_photon_dv, C_photon_dv, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		saving_dir = config_experiment+'_photon_{}_{}'.format(N1, N2)
		utils.save_obj(u_photon_dv, 'u_photon_dv', saving_dir)
		utils.save_obj(eta_0_photon_dv, 'eta_0_photon_dv', saving_dir)
		utils.save_obj(eta_photon_dv, 'eta_photon_dv', saving_dir)
		utils.save_obj(lambda_smoothing_photon_dv, 'lambda_smoothing_photon_dv', saving_dir)
		utils.save_obj(auto_param_obj_history_photon_dv, 'auto_param_obj_history_photon_dv', saving_dir)
		utils.save_obj(auto_param_relaxed_obj_history_photon_dv, 'auto_param_relaxed_obj_history_photon_dv', saving_dir)

		end = time.time()
		print('\n Photon DVC Solution Computed. Time elapsed:', end - start)


	#######################################
	#Solution computation, proton-modality
	#######################################
	if compute_proton == 'yes':
		start = time.time()
		#Compute initial guess, protoni-modality
		# Rx = 190#80#190#190 160 120 80
		# LHS1 = T_list_proton[0]
		# LHS2 = T_list_proton[1]
		# RHS1 = np.array([Rx/np.sum(N)]*LHS1.shape[0])
		# RHS2 = np.array([Rx/np.sum(N)]*LHS2.shape[0])

		# u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x
		# u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x

		# u_init11 = np.concatenate([u1_guess, u2_guess])
		# u_init11 = np.concatenate([u_conv, np.zeros(u2_guess.shape[0])])

		#Initalize parameters
		eta_0 =  (1/(2*np.max(B_proton)))*eta0_coef_proton#0.9 #Initialize eta_0
		eta = np.array([eta_0/len(H_proton)]*len(H_proton))*eta_coef_proton#1e-7
		lambda_smoothing_init = 0 #we don't need smoothing for protons  #np.copy(lambda_smoothing)
		#Set up smoothing matrix
		len_voxels = data['Aphoton'].shape[0]
		beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		S = utils.construct_smoothing_matrix(beams, eps = 5)
		S = S.toarray()
		StS = S.T.dot(S)
		# lambda_smoothing = 1e5#1e7#1e-3 #1e-2

		#Compute the solution:
		
		if precomputed_solution == 'no':
			print('\nComputing the solution')
			#First, compute the solution without dv constraint, protoni-modality
			u_proton_smoothed, eta_0_proton_smoothed, eta_proton_smoothed, lambda_smoothing_proton_smoothed, auto_param_obj_history_proton_smoothed, auto_param_relaxed_obj_history_proton_smoothed = optimization_tools.solver_auto_param(u2_guess, 
				utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_proton, H_proton, alpha_proton, gamma_proton, B_proton, D_proton, C_proton, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0,  proton_only = True)
			saving_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			utils.save_obj(u_proton_smoothed, 'u_proton_smoothed', saving_dir)
			utils.save_obj(eta_0_proton_smoothed, 'eta_0_proton_smoothed', saving_dir)
			utils.save_obj(eta_proton_smoothed, 'eta_proton_smoothed', saving_dir)
			utils.save_obj(lambda_smoothing_proton_smoothed, 'lambda_smoothing_proton_smoothed', saving_dir)
			utils.save_obj(auto_param_obj_history_proton_smoothed, 'auto_param_obj_history_proton_smoothed', saving_dir)
			utils.save_obj(auto_param_relaxed_obj_history_proton_smoothed, 'auto_param_relaxed_obj_history_proton_smoothed', saving_dir)

		if precomputed_solution == 'yes':
			print('\nLoading the solution')
			loading_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
			u_proton_smoothed = utils.load_obj('u_proton_smoothed', loading_dir)
			eta_0_proton_smoothed = utils.load_obj('eta_0_proton_smoothed', loading_dir)
			eta_proton_smoothed = utils.load_obj('eta_proton_smoothed', loading_dir)
			# lambda_smoothing_proton_smoothed = 1e12
			lambda_smoothing_proton_smoothed = utils.load_obj('lambda_smoothing_proton_smoothed', loading_dir)
			#Load lambda smoothing here too
			#Try with all the same parameters and initialize larger smoothing

		end = time.time()
		print('\n Proton without DVC Solution Computed. Time elapsed:', end - start)


		#Now with DVH constraints, protoni-modality
		oar_indices, T_list_proton_dv, T_proton_dv, H_proton_dv, alpha_proton_dv, gamma_proton_dv, B_proton_dv, D_proton_dv, C_proton_dv = utils.generate_dose_volume_input(T_list_proton_max, T_proton_max, H_proton_max, alpha_proton_max, gamma_proton_max, B_proton_max, D_proton_max, C_proton_max, u_proton_smoothed, N, data, Alpha, Beta, Gamma, Delta)

		eta_0 =  (1/(2*np.max(B_proton_dv)))*eta0_coef_proton #Initialize eta_0
		eta = np.array([eta_0/len(H_proton_dv)]*len(H_proton_dv))*eta_coef_proton

		#Update parameters
		if update_parameters == 'yes':
			lambda_smoothing_init = np.copy(lambda_smoothing_proton_smoothed)
			eta_0 = eta_0_proton_smoothed
			eta = utils.update_dose_volume_eta(eta, eta_proton_smoothed, oar_indices, data)
		# lambda_smoothing = 1e5

		u_proton_dv, eta_0_proton_dv, eta_proton_dv, lambda_smoothing_proton_dv, auto_param_obj_history_proton_dv, auto_param_relaxed_obj_history_proton_dv = optimization_tools.solver_auto_param(u_proton_smoothed, 
			utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_proton_dv, H_proton_dv, alpha_proton_dv, gamma_proton_dv, B_proton_dv, D_proton_dv, C_proton_dv, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0,  proton_only = True)
		saving_dir = config_experiment+'_proton_{}_{}'.format(N1, N2)
		utils.save_obj(u_proton_dv, 'u_proton_dv', saving_dir)
		utils.save_obj(eta_0_proton_dv, 'eta_0_proton_dv', saving_dir)
		utils.save_obj(eta_proton_dv, 'eta_proton_dv', saving_dir)
		utils.save_obj(lambda_smoothing_proton_dv, 'lambda_smoothing_proton_dv', saving_dir)
		utils.save_obj(auto_param_obj_history_proton_dv, 'auto_param_obj_history_proton_dv', saving_dir)
		utils.save_obj(auto_param_relaxed_obj_history_proton_dv, 'auto_param_relaxed_obj_history_proton_dv', saving_dir)

		end = time.time()
		print('\n Proton DVC Solution Computed. Time elapsed:', end - start)




