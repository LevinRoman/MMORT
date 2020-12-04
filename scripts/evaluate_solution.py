"""Script to evaluate and visualize treatment plan
Load u and plot the beams as well as contour plots
Create the DVH for BE and dose
ARGS: 
	data_path
	coefficients_path (for Alpha, Beta, Gamma, Delta)
	N1
	N2
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
	# parser.add_argument('--smoothing_ratio', default = 2.0, type = float)
	# parser.add_argument('--lambda_smoothing', default = 1e5, type = float)
	# parser.add_argument('--precomputed_input', default = 'no', type = str)
	parser.add_argument('--N1', default = 43.0, type = float)
	parser.add_argument('--N2', default = 1.0, type = float)
	parser.add_argument('--N_photon', default = 44.0, type = float)
	parser.add_argument('--N_proton', default = 44.0, type = float)
	parser.add_argument('--compute_mult', default = 'no', type = str)
	parser.add_argument('--compute_photon', default = 'no', type = str)
	parser.add_argument('--compute_proton', default = 'no', type = str)
	# parser.add_argument('--Rx', default = 190.0, type = float)
	# parser.add_argument('--eta0_coef_mult', default = 0.9, type = float)
	# parser.add_argument('--eta_coef_mult', default = 1e-7, type = float)
	# parser.add_argument('--eta0_coef_photon', default = 0.9, type = float)
	# parser.add_argument('--eta_coef_photon', default = 1e-7, type = float)
	# parser.add_argument('--eta0_coef_proton', default = 0.9, type = float)
	# parser.add_argument('--eta_coef_proton', default = 1e-7, type = float)
	# parser.add_argument('--eta_step', default = 0.1, type = float)
	# parser.add_argument('--ftol', default = 1e-3, type = float)
	# parser.add_argument('--max_iter', default = 50.0, type = float)
	# parser.add_argument('--update_parameters', default = 'no', type = str)

	# eta_step = 0.1, ftol = 1e-3, max_iter = 50, verbose = 1


	args = parser.parse_args()


	data_name = args.data_name
	config_experiment = args.config_experiment
	# smoothing_ratio = args.smoothing_ratio
	# lambda_smoothing = args.lambda_smoothing
	# precomputed_input = args.precomputed_input
	N1 = args.N1
	N2 = args.N2
	N_photon = args.N_photon
	N_proton = args.N_proton
	N = np.array([N1, N2])
	compute_mult = args.compute_mult
	compute_photon = args.compute_photon
	compute_proton = args.compute_proton
	# Rx = args.Rx
	# eta0_coef_mult = args.eta0_coef_mult
	# eta_coef_mult = args.eta_coef_mult
	# eta0_coef_photon = args.eta0_coef_photon
	# eta_coef_photon = args.eta_coef_photon
	# eta0_coef_proton = args.eta0_coef_proton
	# eta_coef_proton = args.eta_coef_proton
	# eta_step = args.eta_step
	# ftol = args.ftol
	# max_iter = args.max_iter
	# update_parameters = args.update_parameters


	# print('Args: data_name={}, config_experiment={}, smoothing_ratio={}, precomputed_input={}, N1={}, N2={}, N_photon={}, N_proton={}'.format(data_name, config_experiment, smoothing_ratio, precomputed_input, N1, N2, N_photon, N_proton))


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


	if compute_mult == 'yes':
		start = time.time()
		# #Initial input, with no dv constraints, multi-modality
		# if precomputed_input == 'no':
		# 	T_list_mult, T_mult, H_mult, alpha_mult, gamma_mult, B_mult, D_mult, C_mult = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data, modality_names)
		# 	saving_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		# 	utils.save_obj(T_list_mult, 'T_list_mult', saving_dir)
		# 	utils.save_obj(T_mult, 'T_mult', saving_dir)
		# 	utils.save_obj(H_mult, 'H_mult', saving_dir)
		# 	utils.save_obj(alpha_mult, 'alpha_mult', saving_dir)
		# 	utils.save_obj(gamma_mult, 'gamma_mult', saving_dir)
		# 	utils.save_obj(B_mult, 'B_mult', saving_dir)
		# 	utils.save_obj(D_mult, 'D_mult', saving_dir)
		# 	utils.save_obj(C_mult, 'C_mult', saving_dir)
		# 	print('\nInitial input, with dv constraint types, multi-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
		loading_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		T_list_mult = utils.load_obj( 'T_list_mult', loading_dir)
		T_mult = utils.load_obj('T_mult', loading_dir)
		H_mult = utils.load_obj('H_mult', loading_dir)
		alpha_mult = utils.load_obj('alpha_mult', loading_dir)
		gamma_mult = utils.load_obj('gamma_mult', loading_dir)
		B_mult = utils.load_obj('B_mult', loading_dir)
		D_mult = utils.load_obj('D_mult', loading_dir)
		C_mult = utils.load_obj('C_mult', loading_dir)
		print('\nInitial input, with no dv constraints, multi-modality loaded from '+loading_dir)

		
		end = time.time()
		print('Time elapsed:', end - start)


		start = time.time()
		#Max Dose for dv constrained organs input, multi-modality
		# if precomputed_input == 'no':
		# 	T_list_mult_max, T_mult_max, H_mult_max, alpha_mult_max, gamma_mult_max, B_mult_max, D_mult_max, C_mult_max = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
		# 	saving_dir = config_experiment+'_mult_max_{}_{}'.format(N1, N2)
		# 	utils.save_obj(T_list_mult_max, 'T_list_mult_max', saving_dir)
		# 	utils.save_obj(T_mult_max, 'T_mult_max', saving_dir)
		# 	utils.save_obj(H_mult_max, 'H_mult_max', saving_dir)
		# 	utils.save_obj(alpha_mult_max, 'alpha_mult_max', saving_dir)
		# 	utils.save_obj(gamma_mult_max, 'gamma_mult_max', saving_dir)
		# 	utils.save_obj(B_mult_max, 'B_mult_max', saving_dir)
		# 	utils.save_obj(D_mult_max, 'D_mult_max', saving_dir)
		# 	utils.save_obj(C_mult_max, 'C_mult_max', saving_dir)
		# 	print('\nMax Dose input for dv constrained organs input, multi-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
		loading_dir = config_experiment+'_mult_max_{}_{}'.format(N1, N2)
		T_list_mult_max = utils.load_obj('T_list_mult_max', loading_dir)
		T_mult_max = utils.load_obj('T_mult_max', loading_dir)
		H_mult_max = utils.load_obj('H_mult_max', loading_dir)
		alpha_mult_max = utils.load_obj('alpha_mult_max', loading_dir)
		gamma_mult_max = utils.load_obj('gamma_mult_max', loading_dir)
		B_mult_max = utils.load_obj('B_mult_max', loading_dir)
		D_mult_max = utils.load_obj('D_mult_max', loading_dir)
		C_mult_max = utils.load_obj('C_mult_max', loading_dir)
		print('\nMax Dose input for dv constrained organs input, multi-modality loaded from '+loading_dir)

		end = time.time()
		print('Time elapsed:', end - start)


	if compute_photon == 'yes':
		start = time.time()
		#Initial input, with dv constraint types, photon-modality
		# if precomputed_input == 'no':
		# 	T_list_photon, T_photon, H_photon, alpha_photon, gamma_photon, B_photon, D_photon, C_photon = experiments.construct_auto_param_solver_input(np.array([N_photon,0]), Alpha, Beta, Gamma, Delta, data, modality_names)
		# 	saving_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
		# 	utils.save_obj(T_list_photon, 'T_list_photon', saving_dir)
		# 	utils.save_obj(T_photon, 'T_photon', saving_dir)
		# 	utils.save_obj(H_photon, 'H_photon', saving_dir)
		# 	utils.save_obj(alpha_photon, 'alpha_photon', saving_dir)
		# 	utils.save_obj(gamma_photon, 'gamma_photon', saving_dir)
		# 	utils.save_obj(B_photon, 'B_photon', saving_dir)
		# 	utils.save_obj(D_photon, 'D_photon', saving_dir)
		# 	utils.save_obj(C_photon, 'C_photon', saving_dir)
		# 	print('\nInitial input, with dv constraint types, photon-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
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
		# if precomputed_input == 'no':
		# 	T_list_photon_max, T_photon_max, H_photon_max, alpha_photon_max, gamma_photon_max, B_photon_max, D_photon_max, C_photon_max = experiments.construct_auto_param_solver_input(np.array([N_photon,0]), Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
		# 	saving_dir = config_experiment+'_photon_max_{}_{}'.format(N_photon, 0)
		# 	utils.save_obj(T_list_photon_max, 'T_list_photon_max', saving_dir)
		# 	utils.save_obj(T_photon_max, 'T_photon_max', saving_dir)
		# 	utils.save_obj(H_photon_max, 'H_photon_max', saving_dir)
		# 	utils.save_obj(alpha_photon_max, 'alpha_photon_max', saving_dir)
		# 	utils.save_obj(gamma_photon_max, 'gamma_photon_max', saving_dir)
		# 	utils.save_obj(B_photon_max, 'B_photon_max', saving_dir)
		# 	utils.save_obj(D_photon_max, 'D_photon_max', saving_dir)
		# 	utils.save_obj(C_photon_max, 'C_photon_max', saving_dir)
		# 	print('\nMax Dose input for dv constrained organs input, photon-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
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
		# if precomputed_input == 'no':
		# 	T_list_proton, T_proton, H_proton, alpha_proton, gamma_proton, B_proton, D_proton, C_proton = experiments.construct_auto_param_solver_input(np.array([0, N_proton]), Alpha, Beta, Gamma, Delta, data, modality_names)
		# 	saving_dir = config_experiment+'_proton_{}_{}'.format(0, N_proton)
		# 	utils.save_obj(T_list_proton, 'T_list_proton', saving_dir)
		# 	utils.save_obj(T_proton, 'T_proton', saving_dir)
		# 	utils.save_obj(H_proton, 'H_proton', saving_dir)
		# 	utils.save_obj(alpha_proton, 'alpha_proton', saving_dir)
		# 	utils.save_obj(gamma_proton, 'gamma_proton', saving_dir)
		# 	utils.save_obj(B_proton, 'B_proton', saving_dir)
		# 	utils.save_obj(D_proton, 'D_proton', saving_dir)
		# 	utils.save_obj(C_proton, 'C_proton', saving_dir)
		# 	print('\nInitial input, with dv constraint types, proton-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
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
		# if precomputed_input == 'no':
		# 	T_list_proton_max, T_proton_max, H_proton_max, alpha_proton_max, gamma_proton_max, B_proton_max, D_proton_max, C_proton_max = experiments.construct_auto_param_solver_input(np.array([N_proton,0]), Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
		# 	saving_dir = config_experiment+'_proton_max_{}_{}'.format(0, N_proton)
		# 	utils.save_obj(T_list_proton_max, 'T_list_proton_max', saving_dir)
		# 	utils.save_obj(T_proton_max, 'T_proton_max', saving_dir)
		# 	utils.save_obj(H_proton_max, 'H_proton_max', saving_dir)
		# 	utils.save_obj(alpha_proton_max, 'alpha_proton_max', saving_dir)
		# 	utils.save_obj(gamma_proton_max, 'gamma_proton_max', saving_dir)
		# 	utils.save_obj(B_proton_max, 'B_proton_max', saving_dir)
		# 	utils.save_obj(D_proton_max, 'D_proton_max', saving_dir)
		# 	utils.save_obj(C_proton_max, 'C_proton_max', saving_dir)
		# 	print('\nMax Dose input for dv constrained organs input, proton-modality saved to '+saving_dir)
		# if precomputed_input == 'yes':
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
	#Solution computation, multi-modality
	#######################################
	if compute_mult == 'yes':
		start = time.time()
		# #Compute initial guess, multi-modality
		# # Rx = 190#80#190#190 160 120 80
		# LHS1 = T_list_mult[0]
		# LHS2 = T_list_mult[1]
		# RHS1 = np.array([Rx/np.sum(N)]*LHS1.shape[0])
		# RHS2 = np.array([Rx/np.sum(N)]*LHS2.shape[0])

		# u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x
		# u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x

		# u_init11 = np.concatenate([u1_guess, u2_guess])
		# # u_init11 = np.concatenate([u_conv, np.zeros(u2_guess.shape[0])])

		# #Initalize parameters
		# eta_0 =  (1/(2*np.max(B_mult)))*eta0_coef_mult#0.9 #Initialize eta_0
		# eta = np.array([eta_0/len(H_mult)]*len(H_mult))*eta_coef_mult#1e-7
		# lambda_smoothing_init = np.copy(lambda_smoothing)
		# #Set up smoothing matrix
		# len_voxels = data['Aphoton'].shape[0]
		# beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		# beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		# S = utils.construct_smoothing_matrix(beams, eps = 5)
		# S = S.toarray()
		# StS = S.T.dot(S)
		# lambda_smoothing = 1e5#1e7#1e-3 #1e-2

		#Compute the solution:
		
		# if precomputed_input == 'no':
		# 	print('\nComputing the solution')
		# 	#First, compute the solution without dv constraint, multi-modality
		# 	u_mult_smoothed, eta_0_mult_smoothed, eta_mult_smoothed, lambda_smoothing_mult_smoothed, auto_param_obj_history_mult_smoothed, auto_param_relaxed_obj_history_mult_smoothed = optimization_tools.solver_auto_param(u_init11, 
		# 		utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_mult, H_mult, alpha_mult, gamma_mult, B_mult, D_mult, C_mult, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		# 	saving_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		# 	utils.save_obj(u_mult_smoothed, 'u_mult_smoothed', saving_dir)
		# 	utils.save_obj(eta_0_mult_smoothed, 'eta_0_mult_smoothed', saving_dir)
		# 	utils.save_obj(eta_mult_smoothed, 'eta_mult_smoothed', saving_dir)
		# 	utils.save_obj(lambda_smoothing_mult_smoothed, 'lambda_smoothing_mult_smoothed', saving_dir)
		# 	utils.save_obj(auto_param_obj_history_mult_smoothed, 'auto_param_obj_history_mult_smoothed', saving_dir)
		# 	utils.save_obj(auto_param_relaxed_obj_history_mult_smoothed, 'auto_param_relaxed_obj_history_mult_smoothed', saving_dir)

		# if precomputed_input == 'yes':
		print('\nLoading the solution')
		loading_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		u_mult_smoothed = utils.load_obj('u_mult_smoothed', loading_dir)
		auto_param_obj_history_mult_smoothed = utils.load_obj('auto_param_obj_history_mult_smoothed', loading_dir)
		auto_param_relaxed_obj_history_mult_smoothed = utils.load_obj('auto_param_relaxed_obj_history_mult_smoothed', loading_dir)
		# eta_0_mult_smoothed = utils.load_obj('eta_0_mult_smoothed', loading_dir)
		# eta_mult_smoothed = utils.load_obj('eta_mult_smoothed', loading_dir)
		# lambda_smoothing_mult_smoothed = 1e12
		# lambda_smoothing_mult_smoothed = utils.load_obj('lambda_smoothing_mult_smoothed', loading_dir)
		#Load lambda smoothing here too
		#Try with all the same parameters and initialize larger smoothing

		end = time.time()
		print('\n Mult without DVC Solution Computed. Time elapsed:', end - start)


		#Now with DVH constraints, multi-modality
		oar_indices, T_list_mult_dv, T_mult_dv, H_mult_dv, alpha_mult_dv, gamma_mult_dv, B_mult_dv, D_mult_dv, C_mult_dv = utils.generate_dose_volume_input(T_list_mult_max, T_mult_max, H_mult_max, alpha_mult_max, gamma_mult_max, B_mult_max, D_mult_max, C_mult_max, u_mult_smoothed, N, data, Alpha, Beta, Gamma, Delta)

		# eta_0 =  (1/(2*np.max(B_mult_dv)))*eta0_coef_mult #Initialize eta_0
		# eta = np.array([eta_0/len(H_mult_dv)]*len(H_mult_dv))*eta_coef_mult

		#Update parameters
		# if update_parameters == 'yes':
			# lambda_smoothing_init = np.copy(lambda_smoothing_mult_smoothed)
			# eta_0 = eta_0_mult_smoothed
			# eta = utils.update_dose_volume_eta(eta, eta_mult_smoothed, oar_indices, data)
		# lambda_smoothing = 1e5

		# u_mult_dv, eta_0_mult_dv, eta_mult_dv, lambda_smoothing_mult_dv, auto_param_obj_history_mult_dv, auto_param_relaxed_obj_history_mult_dv = optimization_tools.solver_auto_param(u_mult_smoothed, 
			# utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_mult_dv, H_mult_dv, alpha_mult_dv, gamma_mult_dv, B_mult_dv, D_mult_dv, C_mult_dv, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		loading_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		u_mult_dv = utils.load_obj('u_mult_dv', loading_dir)
		auto_param_obj_history_mult_dv = utils.load_obj('auto_param_obj_history_mult_dv', loading_dir)
		auto_param_relaxed_obj_history_mult_dv = utils.load_obj('auto_param_relaxed_obj_history_mult_dv', loading_dir)

		# utils.save_obj(eta_0_mult_dv, 'eta_0_mult_dv', saving_dir)
		# utils.save_obj(eta_mult_dv, 'eta_mult_dv', saving_dir)
		# utils.save_obj(lambda_smoothing_mult_dv, 'lambda_smoothing_mult_dv', saving_dir)
		
		end = time.time()
		print('\n Mult DVC Solution Computed. Time elapsed:', end - start)


		#######################
		#Visualization
		#######################
		saving_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)

		#Figure 1: Constraints 
		plt.figure()
		sns.set()
		plt.plot(optimization_tools.constraints_all(u_mult_dv, H_mult_dv, gamma_mult_dv, D_mult_dv, C_mult_dv, tol = 0.05, verbose = 0)['Constr at u_opt'])
		plt.plot(optimization_tools.constraints_all(u_mult_dv, H_mult_dv, gamma_mult_dv, D_mult_dv, C_mult_dv, tol = 0.05, verbose = 0)['actual constr'], '-.')
		plt.title('OAR constraints')
		plt.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'constraints.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 2: Objective
		fig, ax = plt.subplots(1, 2)
		ax[0].plot(np.concatenate(auto_param_obj_history_mult_dv))
		ax[0].set_title('Objective (Tumor BE')
		ax[1].plot(np.concatenate(auto_param_relaxed_obj_history_mult_dv))
		ax[1].set_title('Relaxed Objective')
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'Objectives.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 3: Beams
		len_voxels = data['Aphoton'].shape[0]
		beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		# beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		fig = plt.figure(figsize = (50, 10))
		for i in range(len(beamlet_indices)):
			ax = fig.add_subplot(150 + i + 1, projection='3d')
			x_beam = data['beamlet_pos'][beamlet_indices[i]][:,0]
			y_beam = data['beamlet_pos'][beamlet_indices[i]][:,1]
			u_beam = u_mult_dv[:data['Aphoton'].shape[1]][beamlet_indices[i]]
			evaluation.plot_beam(ax, x_beam, y_beam, u_beam)
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'beams.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 4: DVH
		fig, axs = plt.subplots(3,2, figsize = (30,30))
		saving_df_path = os.path.abspath(os.path.join('obj', saving_dir))
		evaluation.evaluation_mult_plot_BE(saving_df_path, axs[0,0], axs[0,1], u_mult_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500)
		evaluation.evaluation_photon_plot_BE(saving_df_path, axs[1,0], axs[1,1], u_mult_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500)
		evaluation.evaluation_proton_plot_BE(saving_df_path, axs[2,0], axs[2,1], u_mult_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 10, resolution = 500, max_dose = 45*0.5, dose_resolution = 500)
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'dvh.png')), dpi = 350, bbox_inches = 'tight')


	if compute_photon == 'yes':
		start = time.time()
		# #Compute initial guess, photoni-modality
		# # Rx = 190#80#190#190 160 120 80
		# LHS1 = T_list_photon[0]
		# LHS2 = T_list_photon[1]
		# RHS1 = np.array([Rx/np.sum(N)]*LHS1.shape[0])
		# RHS2 = np.array([Rx/np.sum(N)]*LHS2.shape[0])

		# u1_guess = scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x
		# u2_guess = scipy.optimize.lsq_linear(LHS2, RHS2, bounds = (0, np.inf), tol=1e-3, lsmr_tol=1e-2, max_iter=30, verbose=1).x

		# u_init11 = np.concatenate([u1_guess, u2_guess])
		# # u_init11 = np.concatenate([u_conv, np.zeros(u2_guess.shape[0])])

		# #Initalize parameters
		# eta_0 =  (1/(2*np.max(B_photon)))*eta0_coef_photon#0.9 #Initialize eta_0
		# eta = np.array([eta_0/len(H_photon)]*len(H_photon))*eta_coef_photon#1e-7
		# lambda_smoothing_init = np.copy(lambda_smoothing)
		# #Set up smoothing matrix
		# len_voxels = data['Aphoton'].shape[0]
		# beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		# beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		# S = utils.construct_smoothing_matrix(beams, eps = 5)
		# S = S.toarray()
		# StS = S.T.dot(S)
		# lambda_smoothing = 1e5#1e7#1e-3 #1e-2

		#Compute the solution:
		
		# if precomputed_input == 'no':
		# 	print('\nComputing the solution')
		# 	#First, compute the solution without dv constraint, photoni-modality
		# 	u_photon_smoothed, eta_0_photon_smoothed, eta_photon_smoothed, lambda_smoothing_photon_smoothed, auto_param_obj_history_photon_smoothed, auto_param_relaxed_obj_history_photon_smoothed = optimization_tools.solver_auto_param(u_init11, 
		# 		utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_photon, H_photon, alpha_photon, gamma_photon, B_photon, D_photon, C_photon, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		# 	saving_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
		# 	utils.save_obj(u_photon_smoothed, 'u_photon_smoothed', saving_dir)
		# 	utils.save_obj(eta_0_photon_smoothed, 'eta_0_photon_smoothed', saving_dir)
		# 	utils.save_obj(eta_photon_smoothed, 'eta_photon_smoothed', saving_dir)
		# 	utils.save_obj(lambda_smoothing_photon_smoothed, 'lambda_smoothing_photon_smoothed', saving_dir)
		# 	utils.save_obj(auto_param_obj_history_photon_smoothed, 'auto_param_obj_history_photon_smoothed', saving_dir)
		# 	utils.save_obj(auto_param_relaxed_obj_history_photon_smoothed, 'auto_param_relaxed_obj_history_photon_smoothed', saving_dir)

		# if precomputed_input == 'yes':
		print('\nLoading the solution')
		loading_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
		u_photon_smoothed = utils.load_obj('u_photon_smoothed', loading_dir)
		auto_param_obj_history_photon_smoothed = utils.load_obj('auto_param_obj_history_photon_smoothed', loading_dir)
		auto_param_relaxed_obj_history_photon_smoothed = utils.load_obj('auto_param_relaxed_obj_history_photon_smoothed', loading_dir)
		# eta_0_photon_smoothed = utils.load_obj('eta_0_photon_smoothed', loading_dir)
		# eta_photon_smoothed = utils.load_obj('eta_photon_smoothed', loading_dir)
		# lambda_smoothing_photon_smoothed = 1e12
		# lambda_smoothing_photon_smoothed = utils.load_obj('lambda_smoothing_photon_smoothed', loading_dir)
		#Load lambda smoothing here too
		#Try with all the same parameters and initialize larger smoothing

		end = time.time()
		print('\n Mult without DVC Solution Computed. Time elapsed:', end - start)


		#Now with DVH constraints, photoni-modality
		oar_indices, T_list_photon_dv, T_photon_dv, H_photon_dv, alpha_photon_dv, gamma_photon_dv, B_photon_dv, D_photon_dv, C_photon_dv = utils.generate_dose_volume_input(T_list_photon_max, T_photon_max, H_photon_max, alpha_photon_max, gamma_photon_max, B_photon_max, D_photon_max, C_photon_max, u_photon_smoothed, N, data, Alpha, Beta, Gamma, Delta, photon_only = True)

		# eta_0 =  (1/(2*np.max(B_photon_dv)))*eta0_coef_photon #Initialize eta_0
		# eta = np.array([eta_0/len(H_photon_dv)]*len(H_photon_dv))*eta_coef_photon

		#Update parameters
		# if update_parameters == 'yes':
			# lambda_smoothing_init = np.copy(lambda_smoothing_photon_smoothed)
			# eta_0 = eta_0_photon_smoothed
			# eta = utils.update_dose_volume_eta(eta, eta_photon_smoothed, oar_indices, data)
		# lambda_smoothing = 1e5

		# u_photon_dv, eta_0_photon_dv, eta_photon_dv, lambda_smoothing_photon_dv, auto_param_obj_history_photon_dv, auto_param_relaxed_obj_history_photon_dv = optimization_tools.solver_auto_param(u_photon_smoothed, 
			# utils.organ_photon_matrix('Target', data), S, StS, lambda_smoothing_init, smoothing_ratio, T_photon_dv, H_photon_dv, alpha_photon_dv, gamma_photon_dv, B_photon_dv, D_photon_dv, C_photon_dv, eta_step = eta_step, ftol = ftol, max_iter = max_iter, verbose = 1, eta = eta, eta_0 = eta_0)
		loading_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
		u_photon_dv = utils.load_obj('u_photon_dv', loading_dir)
		auto_param_obj_history_photon_dv = utils.load_obj('auto_param_obj_history_photon_dv', loading_dir)
		auto_param_relaxed_obj_history_photon_dv = utils.load_obj('auto_param_relaxed_obj_history_photon_dv', loading_dir)

		# utils.save_obj(eta_0_photon_dv, 'eta_0_photon_dv', saving_dir)
		# utils.save_obj(eta_photon_dv, 'eta_photon_dv', saving_dir)
		# utils.save_obj(lambda_smoothing_photon_dv, 'lambda_smoothing_photon_dv', saving_dir)
		
		end = time.time()
		print('\n Mult DVC Solution Computed. Time elapsed:', end - start)


		#######################
		#Visualization
		#######################
		saving_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)

		#Figure 1: Constraints 
		plt.figure()
		sns.set()
		plt.plot(optimization_tools.constraints_all(u_photon_dv, H_photon_dv, gamma_photon_dv, D_photon_dv, C_photon_dv, tol = 0.05, verbose = 0)['Constr at u_opt'])
		plt.plot(optimization_tools.constraints_all(u_photon_dv, H_photon_dv, gamma_photon_dv, D_photon_dv, C_photon_dv, tol = 0.05, verbose = 0)['actual constr'], '-.')
		plt.title('OAR constraints')
		plt.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'constraints.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 2: Objective
		fig, ax = plt.subplots(1, 2)
		ax[0].plot(np.concatenate(auto_param_obj_history_photon_dv))
		ax[0].set_title('Objective (Tumor BE')
		ax[1].plot(np.concatenate(auto_param_relaxed_obj_history_photon_dv))
		ax[1].set_title('Relaxed Objective')
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'Objectives.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 3: Beams
		len_voxels = data['Aphoton'].shape[0]
		beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
		# beams = [data['beamlet_pos'][i] for i in beamlet_indices]
		fig = plt.figure(figsize = (50, 10))
		for i in range(len(beamlet_indices)):
			ax = fig.add_subplot(150 + i + 1, projection='3d')
			x_beam = data['beamlet_pos'][beamlet_indices[i]][:,0]
			y_beam = data['beamlet_pos'][beamlet_indices[i]][:,1]
			u_beam = u_photon_dv[:data['Aphoton'].shape[1]][beamlet_indices[i]]
			evaluation.plot_beam(ax, x_beam, y_beam, u_beam)
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'beams.png')), dpi = 350, bbox_inches = 'tight')

		#Figure 4: DVH
		fig, axs = plt.subplots(1,2, figsize = (30,30))
		saving_df_path = os.path.abspath(os.path.join('obj', saving_dir))
		# evaluation.evaluation_photon_plot_BE(saving_df_path, axs[0,0], axs[0,1], u_photon_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500)
		evaluation.evaluation_photon_plot_BE(saving_df_path, axs[0], axs[1], u_photon_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500)
		# evaluation.evaluation_proton_plot_BE(saving_df_path, axs[2,0], axs[2,1], u_photon_dv, N, data, Alpha, Beta, Gamma, Delta, max_BE = 10, resolution = 500, max_dose = 45*0.5, dose_resolution = 500)
		fig.savefig(os.path.abspath(os.path.join('obj', saving_dir, 'dvh.png')), dpi = 350, bbox_inches = 'tight')

