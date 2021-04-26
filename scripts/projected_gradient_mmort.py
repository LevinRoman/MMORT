import argparse
from comet_ml import Experiment
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
#Import mmort modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'mmort')))
import experiments
# import optimization_tools
# import evaluation
import utils
from config import configurations
import torch
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser(description='MMORT')
parser.add_argument('--config_experiment', default = 'Experiment_1', type = str, help = 'Which experiment to run (Options: Experiment_1, Experiment_2). See config file for details')
parser.add_argument('--lr', default=0.1, type=float, help='Lr for Adam or SGD')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--u_max', default=1000, type=float, help='Upper bound on u')
parser.add_argument('--lambda_init', default=1e5, type=float, help='Initial value for lambda')
parser.add_argument('--data_name', default = 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat', type = str)
parser.add_argument('--precomputed', action='store_true', help='Use precomputed smoothed u for DVC initial guess')
parser.add_argument('--initial_guess_for_dv', action='store_true', help='use initial guess for dvc solution')
parser.add_argument('--save_dir', default = 'save_dir', type = str)
parser.add_argument('--optimizer', default = 'Adam', type = str, help='Which optimizer to use (SGD, Adam, LBFGS)')
def relaxed_loss(epoch, u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, device = 'cuda', lambdas = None):
	num_violated = 0
	alpha, beta = radbio_dict['Target'] #Linear and quadratic coefficients
	T = dose_deposition_dict['Target']
	tumor_dose = T@u
	#tumor BE: division to compute average BE, to be on the same scale with max dose
	loss = -N*(alpha*tumor_dose.sum() + beta*(tumor_dose**2).sum())/T.shape[0]
	objective = loss.item()
	experiment.log_metric("Objective", objective, step=epoch)
	OAR_names = list(dose_deposition_dict.keys())[1:] #Not including Target
	#Create penalties and add to the loss
	for oar in OAR_names:
		H = dose_deposition_dict[oar]
		oar_dose = H@u
		constraint_type, constraint_dose, constraint_N = constraint_dict[oar]
		constraint_dose = constraint_dose/constraint_N #Dose per fraction
		gamma, delta = radbio_dict[oar]
		if constraint_type == 'max_dose':
			max_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			max_constr = N*(gamma*oar_dose + delta*oar_dose**2)
			num_violated += ((max_constr - max_constraint_BE)/max_constraint_BE > 0.05).sum()
			if oar in lambdas:
				loss += lambdas[oar]@F.relu(max_constr - max_constraint_BE)**2
			else:
				lambdas[oar] = torch.ones(max_constr.shape[0]).to(device)*args.lambda_init
				loss += lambdas[oar]@F.relu(max_constr - max_constraint_BE)**2
		if constraint_type == 'mean_dose':
			#Mean constr BE across voxels:
			mean_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			#Mean BE across voxels
			mean_constr = N*(gamma*oar_dose.sum() + delta*(oar_dose**2).sum())/H.shape[0]
			num_violated += ((mean_constr - mean_constraint_BE) > 0).sum()
			if oar in lambdas:
				loss += lambdas[oar]*F.relu(mean_constr - mean_constraint_BE)**2
			else:
				lambdas[oar] = args.lambda_init
				loss += lambdas[oar]*F.relu(mean_constr - mean_constraint_BE)**2
	#smoothing constraint:
	experiment.log_metric("Num violated", num_violated, step=epoch)
	smoothing_constr = S@u
	num_violated_smoothing = (smoothing_constr > 0).sum()
	avg_smoothing_violation = (smoothing_constr[smoothing_constr > 0]).mean()
	if 'smoothing' in lambdas:
		loss += lambdas['smoothing']@F.relu(smoothing_constr)**2
	else:
		lambdas['smoothing'] = torch.ones(S.shape[0]).to(device)*args.lambda_init
		loss += lambdas['smoothing']@F.relu(smoothing_constr)**2
	experiment.log_metric("Num violated smoothing", num_violated_smoothing, step=epoch)
	experiment.log_metric("Avg violation smoothing", avg_smoothing_violation, step=epoch)
	experiment.log_metric("Loss", loss.item(), step=epoch)
	return loss, lambdas, num_violated, num_violated_smoothing, objective



def create_coefficient_dicts(data, device):
	"""So far only creates coefficients for the first modality"""
	dose_deposition_dict = {}
	constraint_dict = {}
	radbio_dict = {}
	# coefficient_dict = {}

	organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
	len_voxels = data['Aphoton'].shape[0]
	#[:-1] because we don't want the last isolated voxel
	organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
	OAR_constr_types = np.squeeze(data['OAR_constraint_types'])
	OAR_constr_values = np.squeeze(data['OAR_constraint_values'])
	for organ_number, organ_name in enumerate(organ_names):
		oar_number = organ_number - 1 #Because Target is also an organ
		# dose_deposition_dict[organ_name] = torch.from_numpy(data['Aphoton'][organ_indices[organ_number]])
		dose_deposition_dict[organ_name] = csr_matrix_to_coo_tensor(data['Aphoton'][organ_indices[organ_number]]).to(device)
		if organ_name == 'Target':
			radbio_dict[organ_name] = Alpha[0], Beta[0] #So far, only photons
			# coefficient_dict[organ_name] = alpha*torch.ones(T.shape[0])@T
		if organ_name != 'Target':
			constraint_type = OAR_constr_types[oar_number].strip()
			constraint_dose = OAR_constr_values[oar_number]
			constraint_N = 44
			constraint_dict[organ_name] = constraint_type, constraint_dose, constraint_N
			radbio_dict[organ_name] = Gamma[oar_number][0], Delta[oar_number][0]
	return dose_deposition_dict, constraint_dict, radbio_dict

def dv_adjust_coefficient_dicts(data, dose_deposition_dict, dv_to_max_oar_ind_dict, device):
	"""So far only creates coefficients for the first modality"""
	organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
	len_voxels = data['Aphoton'].shape[0]
	#[:-1] because we don't want the last isolated voxel
	organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
	for organ_number, organ_name in enumerate(organ_names):
		if organ_name in dv_to_max_oar_ind_dict:
			print('\n Old len:', dose_deposition_dict[organ_name].shape[0])
			dose_matrix = data['Aphoton'][organ_indices[organ_number]][dv_to_max_oar_ind_dict[organ_name]]
			dose_deposition_dict[organ_name] = csr_matrix_to_coo_tensor(dose_matrix).to(device)
			print('\n New len:', dose_deposition_dict[organ_name].shape[0])
	return dose_deposition_dict

def csr_matrix_to_coo_tensor(matrix):
	coo = scipy.sparse.coo_matrix(matrix)

	values = coo.data
	indices = np.vstack((coo.row, coo.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo.shape

	return torch.sparse.FloatTensor(i, v, torch.Size(shape))

if __name__ == '__main__':
	args = parser.parse_args()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	experiment = Experiment(api_key='P63wSM91MmVDh80ZBZbcylZ8L', project_name='mmort_torch')

	#########################
	##Load raw data
	#########################
	data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', args.data_name))
	data = scipy.io.loadmat(data_path)
	# Alpha = np.array([0.35, 0.35])
	# Beta = np.array([0.175, 0.175])
	# Gamma = np.array([np.array([0.35, 0.35]),
	# 				  np.array([0.35, 0.35]),
	# 				  np.array([0.35, 0.35]),
	# 				  np.array([0.35, 0.35]),
	# 				  np.array([0.35, 0.35])               
	# 				 ])
	# Delta = np.array([np.array([0.07, 0.07]),
	# 				  np.array([0.07, 0.07]),
	# 				  np.array([0.175, 0.175]),
	# 				  np.array([0.175, 0.175]),
	# 				  np.array([0.175, 0.175])                
	# 				 ])
	# modality_names = np.array(['Aphoton', 'Aproton'])

	###########################
	#Experimental Setup Part
	###########################

	#Load experimental setup from config
	experiment_setup = configurations[args.config_experiment]

	Alpha = experiment_setup['Alpha']
	Beta = experiment_setup['Beta']
	Gamma = experiment_setup['Gamma']
	Delta = experiment_setup['Delta']

	modality_names = experiment_setup['modality_names']

	print('\nExperimental Setup: \nAlpha={} \nBeta={} \nGamma={} \nDelta={} \nModality Names: {}'.format(Alpha, Beta, Gamma, Delta, modality_names))

	num_body_voxels = 683189
	data['Aphoton'][-1] = data['Aphoton'][-1]/num_body_voxels
	data['Aproton'][-1] = data['Aproton'][-1]/num_body_voxels

	for modality in modality_names:
	        data[modality] = scipy.sparse.csr_matrix(data[modality])

	#Data with max dose (to be used with DVC):
	data_max_dose = copy.deepcopy(data)
	data_max_dose['OAR_constraint_types'][data_max_dose['OAR_constraint_types'] == 'dose_volume'] = 'max_dose'

	print('\nData loaded from '+data_path)
	
	
	
	###################################
	##Solution: photons
	###################################
	#No dose volume
	N = 44
	#Set up smoothing matrix
	len_voxels = data['Aphoton'].shape[0]
	beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
	beams = [data['beamlet_pos'][i] for i in beamlet_indices]
	S = csr_matrix_to_coo_tensor(utils.construct_smoothing_matrix_relative(beams, 0.25, eps = 5)).to(device)

	dose_deposition_dict, constraint_dict, radbio_dict = create_coefficient_dicts(data, device)

	print('\nDose_deposition_dict:', dose_deposition_dict)
	print('\nConstraint dict:', constraint_dict)
	print('\nradbio_dict:', radbio_dict)
	print('\nN:', N)
	print('\nS shape:', S.shape)

	#Setup optimization
	if not args.precomputed:
		print('\n Running optimization...')
		Rx = 81
		print(data['num_voxels'][0])
		LHS1 = data['Aphoton'][:np.squeeze(data['num_voxels'])[0]]
		RHS1 = np.array([Rx/N]*LHS1.shape[0])
		u = torch.Tensor(scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-4, max_iter=100, verbose=1).x)
		u = u.to(device)
		u.requires_grad_()

		if args.optimizer == 'SGD':
			optimizer = optim.SGD([u], lr=args.lr, momentum=0.9, nesterov = True)
		elif args.optimizer == 'Adam':
			optimizer = optim.Adam([u], lr=args.lr)
		elif args.optimizer == 'LBFGS':
			optimizer = optim.LBFGS([u])

		lambdas = {}
		for epoch in range(args.num_epochs):
			optimizer.zero_grad()
			loss, lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss(epoch, u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, device = device, lambdas = lambdas)
			print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
			loss.backward()
			optimizer.step()
			#Box constraint
			u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))

		print(u)

		#To run:  python3 projected_gradient_mmort.py --lr 1e-6 --lambda_init 1e3 --num_epochs 10000
		utils.save_obj(u.detach().cpu().numpy(), 'u_photon_pytorch')

	if args.precomputed:
		  u = torch.from_numpy(utils.load_obj('u_photon_pytorch', ''))
		  u = u.to(device)
		  u.requires_grad_()
	####################
	##DVC: Photons
	####################
	print('Setting up DVC data...')
	#Setup input
	oar_indices, dv_to_max_oar_ind_dict = utils.generate_dose_volume_input_torch(u.detach().cpu().numpy(), np.array([N, 0]), data, Alpha, Beta, Gamma, Delta, photon_only = True, proton_only = False)

	dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv = create_coefficient_dicts(data_max_dose, device)
	
	# for organ in dv_to_max_oar_ind_dict:
	# 	print('\n DVC organ {} with constr: {}'.format(organ, constraint_dict_dv[organ]))
	# 	print('\n Old len:', dose_deposition_dict_dv[organ].shape[0])
	# 	dose_deposition_dict_dv[organ] = dose_deposition_dict_dv[organ][torch.from_numpy(dv_to_max_oar_ind_dict[organ]).to(device)]
	# 	print('\n New len:', dose_deposition_dict_dv[organ].shape[0])
	dose_deposition_dict_dv = dv_adjust_coefficient_dicts(data_max_dose, dose_deposition_dict_dv, dv_to_max_oar_ind_dict, device)

	#Compute solution
	print('Computing DV solution')
	print('\nDose_deposition_dict:', dose_deposition_dict_dv)
	print('\nConstraint dict:', constraint_dict_dv)
	print('\nradbio_dict:', radbio_dict_dv)
	print('\nN:', N)
	print('\nS shape:', S.shape)


	#Setup optimization
	print('\n Running optimization...')
	#Uncomment this to setup an initial guess on u from scratch
	if args.initial_guess_for_dv:
		Rx = 81
		print(data['num_voxels'][0])
		LHS1 = data['Aphoton'][:np.squeeze(data['num_voxels'])[0]]
		RHS1 = np.array([Rx/N]*LHS1.shape[0])
		u = torch.Tensor(scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-4, max_iter=100, verbose=1).x)
		u = u.to(device)
		u.requires_grad_()

		if args.optimizer == 'SGD':
			optimizer = optim.SGD([u], lr=args.lr, momentum=0.9, nesterov = True)
		elif args.optimizer == 'Adam':
			optimizer = optim.Adam([u], lr=args.lr)
		elif args.optimizer == 'LBFGS':
			optimizer = optim.LBFGS([u])

	lambdas = {dv_organ: args.lambda_init/10 for dv_organ in dv_to_max_oar_ind_dict}#{}
	for epoch in range(args.num_epochs):
		optimizer.zero_grad()
		loss, lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss(epoch, u, N, dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv, S, experiment, device = device, lambdas = lambdas)
		print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
		loss.backward()
		optimizer.step()
		#Box constraint
		u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))

	print(u)

	#To run:  python3 projected_gradient_mmort.py --lr 1e-6 --lambda_init 1e3 --num_epochs 10000
	utils.save_obj(u.detach().cpu().numpy(), 'u_photon_dv_pytorch', args.save_dir)
	#
	#TODO:
	#dvh
	#multi-modality
	#lagrange optimization
	#IMRT
