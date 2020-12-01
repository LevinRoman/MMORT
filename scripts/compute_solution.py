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
	parser = argparse.ArgumentParser()



	parser.add_argument('--data_name', default = 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat', type = str)
	parser.add_argument('--config_experiment', default = 'Experiment_1', type = str)
	parser.add_argument('--smoothing_ratio', default = 2.0, type = float)
	parser.add_argument('--precomputed_input', default = 'no', type = str)
	parser.add_argument('--N1', default = 43.0, type = float)
	parser.add_argument('--N2', default = 1.0, type = float)
	parser.add_argument('--N_photon', default = 44.0, type = float)
	parser.add_argument('--N_proton', default = 44.0, type = float)



	args = parser.parse_args()


	data_name = args.data_name
	config_experiment = args.config_experiment
	smoothing_ratio = args.smoothing_ratio
	precomputed_input = args.precomputed_input
	N1 = args.N1
	N2 = args.N2
	N_photon = args.N_photon
	N_proton = args.N_proton
	N = np.array([N1, N2])

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

	#Initial input, with dv constraint types, multi-modality
	if precomputed_input == 'no':
		T_list_mult, T_mult, H_mult, alpha_mult, gamma_mult, B_mult, D_mult, C_mult = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data, modality_names)
		saving_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		utils.save_obj(T_list_mult, 'T_list_mult', saving_dir)
		utils.save_obj(T_mult, 'T_mult', saving_dir)
		utils.save_obj(H_mult, 'H_mult', saving_dir)
		utils.save_obj(alpha_mult, 'alpha_mult', saving_dir)
		utils.save_obj(gamma_mult, 'gamma_mult', saving_dir)
		utils.save_obj(B_mult, 'B_mult', saving_dir)
		utils.save_obj(D_mult, 'D_mult', saving_dir)
		utils.save_obj(C_mult, 'C_mult', saving_dir)
		print('\nInitial input, with dv constraint types, multi-modality saved to '+saving_dir)
	if precomputed_input == 'yes':
		loading_dir = config_experiment+'_mult_{}_{}'.format(N1, N2)
		T_list_mult = utils.load_obj( 'T_list_mult', loading_dir)
		T_mult = utils.load_obj('T_mult', loading_dir)
		H_mult = utils.load_obj('H_mult', loading_dir)
		alpha_mult = utils.load_obj('alpha_mult', loading_dir)
		gamma_mult = utils.load_obj('gamma_mult', loading_dir)
		B_mult = utils.load_obj('B_mult', loading_dir)
		D_mult = utils.load_obj('D_mult', loading_dir)
		C_mult = utils.load_obj('C_mult', loading_dir)
		print('\nInitial input, with dv constraint types, multi-modality loaded from '+loading_dir)


	#Max Dose for dv constrained organs input, multi-modality
	if precomputed_input == 'no':
		T_list_mult_max, T_mult_max, H_mult_max, alpha_mult_max, gamma_mult_max, B_mult_max, D_mult_max, C_mult_max = experiments.construct_auto_param_solver_input(N, Alpha, Beta, Gamma, Delta, data_max_dose, modality_names)
		saving_dir = config_experiment+'_mult_max_{}_{}'.format(N1, N2)
		utils.save_obj(T_list_mult_max, 'T_list_mult_max', saving_dir)
		utils.save_obj(T_mult_max, 'T_mult_max', saving_dir)
		utils.save_obj(H_mult_max, 'H_mult_max', saving_dir)
		utils.save_obj(alpha_mult_max, 'alpha_mult_max', saving_dir)
		utils.save_obj(gamma_mult_max, 'gamma_mult_max', saving_dir)
		utils.save_obj(B_mult_max, 'B_mult_max', saving_dir)
		utils.save_obj(D_mult_max, 'D_mult_max', saving_dir)
		utils.save_obj(C_mult_max, 'C_mult_max', saving_dir)
		print('\nMax Dose input for dv constrained organs input, multi-modality saved to '+saving_dir)
	if precomputed_input == 'yes':
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











