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
parser.add_argument('--lr', default=1e-3, type=float, help='Lr for Adam or SGD')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--u_max', default=1000, type=float, help='Upper bound on u')
parser.add_argument('--lambda_init', default=1e5, type=float, help='Initial value for lambda')
parser.add_argument('--data_name', default = 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat', type = str)
parser.add_argument('--precomputed', action='store_true', help='Use precomputed smoothed u for DVC initial guess')
parser.add_argument('--initial_guess_for_dv', action='store_true', help='use initial guess for dvc solution')
parser.add_argument('--save_dir', default = 'save_dir', type = str)
parser.add_argument('--optimizer', default = 'Adam', type = str, help='Which optimizer to use (SGD, Adam, LBFGS)')
#Lagnragian optimization args
parser.add_argument('--lagrange', action='store_true', help='use Lagrangian optimization: do alternating grad desc wrt u and ascent wrt lamda')
parser.add_argument('--lambda_lr', default=1e-3, type=float, help='Lr for Adam or SGD for Lambda lagrange update')
#N optimization args
parser.add_argument('--optimize_N', action='store_true', help='Attempt N optimization')
parser.add_argument('--tumor_double_time', default=10.0, type=float, help='Tumor doubling time')
parser.add_argument('--N_max', default=50.0, type=float, help='Max fractionation')
parser.add_argument('--N_lr', default=0.1, type=float, help='Lr for Adam or SGD for N update')
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
	avg_smoothing_violation = (smoothing_constr[smoothing_constr > 0]).max()
	if 'smoothing' in lambdas:
		loss += lambdas['smoothing']@F.relu(smoothing_constr)**2
	else:
		lambdas['smoothing'] = torch.ones(S.shape[0]).to(device)*args.lambda_init
		loss += lambdas['smoothing']@F.relu(smoothing_constr)**2
	experiment.log_metric("Num violated smoothing", num_violated_smoothing, step=epoch)
	experiment.log_metric("Max violation smoothing", avg_smoothing_violation, step=epoch)
	experiment.log_metric("Loss", loss.item(), step=epoch)
	return loss, lambdas, num_violated, num_violated_smoothing, objective

def relaxed_loss_lagrange(epoch, u, lambdas_var, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, args, device = 'cuda', lambdas = None):
	"""
	Lambdas_var is a list of lambda variable tensors correponding to the OAR_names
	"""
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
	for oar_num, oar in enumerate(OAR_names):
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
				loss += lambdas_var[oar_num]@(max_constr - max_constraint_BE)
			else:
				raise ValueError('Lambdas cannot be None for Lagrangian optimization')
		if constraint_type == 'mean_dose':
			#Mean constr BE across voxels:
			mean_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			#Mean BE across voxels
			mean_constr = N*(gamma*oar_dose.sum() + delta*(oar_dose**2).sum())/H.shape[0]
			num_violated += ((mean_constr - mean_constraint_BE) > 0).sum()
			if oar in lambdas:
				loss += lambdas_var[oar_num]*(mean_constr - mean_constraint_BE)
			else:
				raise ValueError('Lambdas cannot be None for Lagrangian optimization')
	#smoothing constraint: should be the last element of lambdas_var
	experiment.log_metric("Num violated", num_violated, step=epoch)
	smoothing_constr = S@u
	num_violated_smoothing = (smoothing_constr > 0).sum()
	max_smoothing_violation = (smoothing_constr[smoothing_constr > 0]).max()
	if 'smoothing' in lambdas:
		loss += lambdas_var[-1]@smoothing_constr
	else:
		raise ValueError('Lambdas cannot be None for Lagrangian optimization')
	experiment.log_metric("Num violated smoothing", num_violated_smoothing.item(), step=epoch)
	experiment.log_metric("Max violation smoothing", max_smoothing_violation.item(), step=epoch)
	experiment.log_metric("Loss", loss.item(), step=epoch)
	experiment.log_metric("Avg lambda", np.mean([lambda_.mean().item() for lambda_ in lambdas_var]), step=epoch)
	experiment.log_metric("Min lambda", np.min([lambda_.min().item() for lambda_ in lambdas_var]), step=epoch)
	experiment.log_metric("Max lambda", np.max([lambda_.max().item() for lambda_ in lambdas_var]), step=epoch)
	experiment.log_metric('Avg u', u.mean().item(), step = epoch)
	if args.optimize_N:
		loss = loss + ((N-1)*np.log(2.0)/args.tumor_double_time)
	return loss, num_violated, num_violated_smoothing, objective

def initialize_lambdas(u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, device = 'cuda'):
	with torch.no_grad():
		lambdas = {}
		num_violated = 0
		alpha, beta = radbio_dict['Target'] #Linear and quadratic coefficients
		T = dose_deposition_dict['Target']
		tumor_dose = T@u
		#tumor BE: division to compute average BE, to be on the same scale with max dose
		loss = -N*(alpha*tumor_dose.sum() + beta*(tumor_dose**2).sum())/T.shape[0]
		objective = loss.item()
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
				
				lambdas[oar] = torch.zeros_like(max_constr - max_constraint_BE).to(device)
			
			if constraint_type == 'mean_dose':
				#Mean constr BE across voxels:
				mean_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
				#Mean BE across voxels
				mean_constr = N*(gamma*oar_dose.sum() + delta*(oar_dose**2).sum())/H.shape[0]
				num_violated += ((mean_constr - mean_constraint_BE) > 0).sum()
				
				lambdas[oar] = torch.zeros_like(mean_constr - mean_constraint_BE).to(device)
				
		#smoothing constraint:

		smoothing_constr = S@u
		
		lambdas['smoothing'] = torch.zeros_like(smoothing_constr).to(device)
	lambdas_var = [lambdas[constr].requires_grad_() for constr in lambdas]
	print('\n Initializing Lambdas:')
	for i, constr in enumerate(lambdas):
		print('Lambdas:', lambdas[constr].shape)
		print('Vars:', lambdas_var[i].shape)
	# raise ValueError('Stop')
	return lambdas, lambdas_var

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
		else:
			raise ValueError('The optimizer option {} is not supported'.format(args.optimizer))

		if not args.lagrange:
			lambdas = {}

		if args.lagrange:
			lambdas, lambdas_var = initialize_lambdas(u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, device = 'cuda')
			# for constraint in lambdas_var:
			# 	lambdas[constraint].requires_grad_()
			optimizer_lambdas = optim.Adam(lambdas_var, lr=args.lambda_lr)

		if not args.lagrange:	
			for epoch in range(args.num_epochs):
				optimizer.zero_grad()
				loss, lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss(epoch, u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, device = device, lambdas = lambdas)
				print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
				loss.backward()
				optimizer.step()
				#Box constraint
				u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))

		if args.lagrange:
			for epoch in range(args.num_epochs):
				#Update u:
				print('\n u step')
				optimizer.zero_grad()
				loss, num_violated, num_violated_smoothing, objective = relaxed_loss_lagrange(epoch, u, lambdas_var, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, args, device = device, lambdas = lambdas)
				print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
				loss.backward()
				optimizer.step()
				#Box constraint
				u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))
				
				#Update lambdas:
				print('\n lambdas step')		
				optimizer_lambdas.zero_grad()
				loss_lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss_lagrange(epoch, u, lambdas_var, N, dose_deposition_dict, constraint_dict, radbio_dict, S, experiment, args, device = device, lambdas = lambdas)
				loss_lambdas = (-1)*loss_lambdas
				loss_lambdas.backward()
				optimizer_lambdas.step()
				#Box contraint (lambda >= 0)
				for constraint in range(len(lambdas_var)):
					lambdas_var[constraint].data = torch.maximum(lambdas_var[constraint], torch.zeros_like(lambdas_var[constraint]))
		print(u)

		#To run:  python3 projected_gradient_mmort.py --lr 1e-6 --lambda_init 1e3 --num_epochs 10000
		if not args.lagrange:
			utils.save_obj(u.detach().cpu().numpy(), 'u_photon_pytorch')
		if args.lagrange:
			utils.save_obj(u.detach().cpu().numpy(), 'u_photon_pytorch_lagrange')

	if args.precomputed:
		if not args.lagrange:
			u = torch.from_numpy(utils.load_obj('u_photon_pytorch', ''))
			u = u.to(device)
			u.requires_grad_()
		if args.lagrange:
			u = torch.from_numpy(utils.load_obj('u_photon_pytorch_lagrange', ''))
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

	if args.optimize_N:
		N = torch.tensor(44.0)
		N.requires_grad_()

	if args.optimizer == 'SGD':
		optimizer = optim.SGD([u], lr=args.lr, momentum=0.9, nesterov = True)
	elif args.optimizer == 'Adam':
		if args.optimize_N:
			optimizer = optim.Adam([{'params': [u]}, 
			{'params': [N], 'lr': args.N_lr}], lr=args.lr)
		else:
			optimizer = optim.Adam([u], lr=args.lr)
	elif args.optimizer == 'LBFGS':
		optimizer = optim.LBFGS([u])
	else:
		raise ValueError('The optimizer option {} is not supported'.format(args.optimizer))

	if not args.lagrange:
		lambdas = {}

	if args.lagrange:
		lambdas, lambdas_var = initialize_lambdas(u, N, dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv, S, experiment, device = 'cuda')
		# for constraint in lambdas:
		# 	lambdas[constraint].requires_grad_()
		optimizer_lambdas = optim.Adam(lambdas_var, lr=args.lambda_lr)

	# lambdas = {dv_organ: torch.ones(dv_to_max_oar_ind_dict[dv_organ].shape[0]).to(device)*args.lambda_init/10 for dv_organ in dv_to_max_oar_ind_dict}#{}
	if not args.lagrange:
		for epoch in range(args.num_epochs):
			optimizer.zero_grad()
			loss, lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss(epoch, u, N, dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv, S, experiment, device = device, lambdas = lambdas)
			print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
			loss.backward()
			optimizer.step()
			#Box constraint
			u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))

	if args.lagrange:
		for epoch in range(args.num_epochs):
			#Update u:
			print('\n u step')
			optimizer.zero_grad()
			loss, num_violated, num_violated_smoothing, objective = relaxed_loss_lagrange(epoch, u, lambdas_var, N, dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv, S, experiment, args, device = device, lambdas = lambdas)
			print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss, objective, num_violated, num_violated_smoothing))
			experiment.log_metric("Loss_u", loss.item(), step=epoch)
			loss.backward()
			optimizer.step()
			#Box constraint
			u.data = torch.maximum(torch.minimum(u, torch.ones_like(u)*args.u_max), torch.zeros_like(u))
			if args.optimize_N:
				N.data = torch.maximum(torch.minimum(N, torch.ones_like(N)*args.N_max), torch.zeros_like(N))
				experiment.log_metric("N", N.item(), step=epoch)
			
			#Update lambdas:
			print('\n lambdas step')
			optimizer_lambdas.zero_grad()
			loss_lambdas, num_violated, num_violated_smoothing, objective = relaxed_loss_lagrange(epoch, u, lambdas_var, N, dose_deposition_dict_dv, constraint_dict_dv, radbio_dict_dv, S, experiment, args, device = device, lambdas = lambdas)
			print('\n Loss {} \n Objective {} \n Num Violated {} \n Num Violated Smoothing {}'.format(loss_lambdas, objective, num_violated, num_violated_smoothing))
			loss_lambdas = (-1)*loss_lambdas
			experiment.log_metric("Loss_l", loss_lambdas.item(), step=epoch)
			loss_lambdas.backward()
			optimizer_lambdas.step()
			#Box contraint (lambda >= 0)
			for constraint in range(len(lambdas_var)):
				lambdas_var[constraint].data = torch.maximum(lambdas_var[constraint], torch.zeros_like(lambdas_var[constraint]))
	print(u)

	#To run:  python3 projected_gradient_mmort.py --lr 1e-6 --lambda_init 1e3 --num_epochs 10000
	if not args.lagrange:
		utils.save_obj(u.detach().cpu().numpy(), 'u_photon_dv_pytorch', args.save_dir)
	if args.lagrange:
		utils.save_obj(u.detach().cpu().numpy(), 'u_photon_dv_pytorch_lagrange', args.save_dir)
	#
	#TODO:
	#dvh + 
	#multi-modality
	#lagrange optimization
	#IMRT
	#N optimization is so far implemented only in the second half
	#Run with faithfully computed init guess for dv