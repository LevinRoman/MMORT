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
parser.add_argument('--lr', default=0.1, type=float, help='Lr for Adam or SGD')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--u_max', default=1000, type=float, help='Upper bound on u')


def relaxed_loss(u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, lambdas = None):
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
			num_violated += ((max_constr - max_constraint_BE) > 0).sum()
			if oar in lambdas:
				loss += lambdas[oar]@F.relu(max_constr - max_constraint_BE)
			else:
				lambdas[oar] = torch.ones(max_constr.shape[0])*1e5
				loss += lambdas[oar]@F.relu(max_constr - max_constraint_BE)
		if constraint_type == 'mean_dose':
			#Mean constr BE across voxels:
			mean_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			#Mean BE across voxels
			mean_constr = N*(gamma*oar_dose.sum() + delta*(oar_dose**2).sum())/H.shape[0]
			num_violated += ((mean_constr - mean_constraint_BE) > 0).sum()
			if oar in lambdas:
				loss += lambdas[oar]*F.relu(mean_constr - mean_constraint_BE)
			else:
				lambdas[oar] = 1e5
				loss += lambdas[oar]*F.relu(mean_constr - mean_constraint_BE)
	#smoothing constraint:
	if 'smoothing' in lambdas:
		loss += lambdas['smoothing']@F.relu(S@u)
	else:
		lambdas['smoothing'] = torch.ones(S.shape[0])*1e5
		loss += lambdas['smoothing']@F.relu(S@u)
	return loss, lambdas, num_violated, objective



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

	data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat'))
	# data_no_body_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'ProstateExample.mat'))
	data = scipy.io.loadmat(data_path)
	# data =  scipy.io.loadmat(data_no_body_path)
	Alpha = np.array([0.35, 0.35])
	Beta = np.array([0.175, 0.175])
	Gamma = np.array([np.array([0.35, 0.35]),
					  np.array([0.35, 0.35]),
					  np.array([0.35, 0.35]),
					  np.array([0.35, 0.35]),
					  np.array([0.35, 0.35])               
					 ])
	Delta = np.array([np.array([0.07, 0.07]),
					  np.array([0.07, 0.07]),
					  np.array([0.175, 0.175]),
					  np.array([0.175, 0.175]),
					  np.array([0.175, 0.175])                
					 ])
	modality_names = np.array(['Aphoton', 'Aproton'])

	num_body_voxels = 683189
	data['Aphoton'][-1] = data['Aphoton'][-1]/num_body_voxels
	data['Aproton'][-1] = data['Aproton'][-1]/num_body_voxels

	for modality in modality_names:
	        data[modality] = scipy.sparse.csr_matrix(data[modality])

	N = 44
	#Set up smoothing matrix
	len_voxels = data['Aphoton'].shape[0]
	beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
	beams = [data['beamlet_pos'][i] for i in beamlet_indices]
	# S = torch.from_numpy(utils.construct_smoothing_matrix_relative(beams, 0.25, eps = 5).todense())
	S = csr_matrix_to_coo_tensor(utils.construct_smoothing_matrix_relative(beams, 0.25, eps = 5)).to(device)

	dose_deposition_dict, constraint_dict, radbio_dict = create_coefficient_dicts(data, device)

	print('\nDose_deposition_dict:', dose_deposition_dict)
	print('\nConstraint dict:', constraint_dict)
	print('\nradbio_dict:', radbio_dict)
	print('\nN:', N)
	print('\nS shape:', S.shape)

	print('\n Running optimization...')
	Rx = 81
	print(data['num_voxels'][0])
	LHS1 = data['Aphoton'][:np.squeeze(data['num_voxels'])[0]]
	RHS1 = np.array([Rx/N]*LHS1.shape[0])
	u = torch.Tensor(scipy.optimize.lsq_linear(LHS1, RHS1, bounds = (0, np.inf), tol=1e-4, lsmr_tol=1e-4, max_iter=100, verbose=1).x)
	u = u.to(device)
	u.requires_grad_()

	
	optimizer = optim.SGD([u], lr=args.lr, momentum=0.9, nesterov = True)

	lambdas = None
	for epoch in range(args.num_epochs):
		optimizer.zero_grad()
		loss, lambdas, num_violated, objective = relaxed_loss(u, N, dose_deposition_dict, constraint_dict, radbio_dict, S, lambdas = lambdas)
		print('\n Loss {} \n Objective {} \n Num Violated {}'.format(loss, objective, num_violated))
		loss.backward()
		optimizer.step()
		#Box constraint
		u.data = torch.max(torch.min(u, 0), args.u_max)

