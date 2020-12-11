import pickle
import pathlib
import os
import scipy.linalg
import scipy.sparse
import scipy.spatial
import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'mmort')))
import experiments
import optimization_tools
import evaluation
import utils
# from config import configurations

def save_obj(obj, name, directory = ''):
	pathlib.Path('obj').mkdir(exist_ok=True)
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



def generate_dose_volume_input(T_list_mult_max, T_mult_max, H_mult_max, alpha_mult_max, gamma_mult_max, B_mult_max, D_mult_max, C_mult_max, u_mult, N, data, Alpha, Beta, Gamma, Delta, photon_only = False, proton_only = False):
	"""Based on u_mult, add max-constrained additional OARs, note that this would not work with mean dose OARs"""
	oar_indices = np.split(np.arange(data['Aphoton'].shape[0]), np.cumsum(np.squeeze(data['num_voxels'])[1:]))[:-1]
	dv_oar_names = [str(i[0]) for i in np.squeeze(data['Organ'])[1:][data['OAR_constraint_types'] == 'dose_volume']]
	for i, name in enumerate(dv_oar_names):
		if photon_only:
			_, _, oar_BE, _, oar_photon_BE = evaluation.evaluation_function_photon(u_mult, N, data, name, Alpha, Beta, Gamma, Delta, 3000, resolution = 500)
			print(oar_BE, oar_photon_BE)
		if proton_only:
			_, _, oar_BE, _, oar_proton_BE = evaluation.evaluation_function_proton(u_mult, N, data, name, Alpha, Beta, Gamma, Delta, 3000, resolution = 500)
			print(oar_BE, oar_proton_BE)
		if (not photon_only) and (not proton_only):
			_,_,_,_,_, oar_photon_BE, oar_proton_BE = evaluation.evaluation_function(u_mult, N, data, name, Alpha, Beta, Gamma, Delta, 3000, resolution = 500)
			oar_BE = oar_photon_BE + oar_proton_BE

		#Take the low 50% of the voxels, should make this more general to handle arbitrary percentage

		cur_oar_number_ = np.arange(len(oar_indices))[data['OAR_constraint_types'] == 'dose_volume'][i]
		print('\n Min and max index before DV constraint:{}{}'.format(np.min(oar_indices[cur_oar_number_]), np.max(oar_indices[cur_oar_number_])))

		cur_oar_indices_to_max_constrain = np.argsort(oar_BE)[:(oar_BE.shape[0]//2 + oar_BE.shape[0]%2)]
		# constraint = np.array(C_mult_max)[oar_indices[0]]
		cur_oar_number = np.arange(len(oar_indices))[data['OAR_constraint_types'] == 'dose_volume'][i]
		oar_indices[cur_oar_number] = oar_indices[cur_oar_number][cur_oar_indices_to_max_constrain]

		print('\n Current DV OAR:{} Number:{}'.format(name, cur_oar_number))
		print('\n Min and max index to be DV constrained:{}{}'.format(np.min(oar_indices[cur_oar_number]), np.max(oar_indices[cur_oar_number])))
		print('\n Min and max in all cur_oar_indices_to_max_constrain:{}{}'.format(np.min(np.argsort(oar_BE)), np.max(np.argsort(oar_BE))))

	updated_C = [C_mult_max[i] for oar in oar_indices for i in oar]
	updated_H = [H_mult_max[i] for oar in oar_indices for i in oar]
	updated_gamma = [gamma_mult_max[i] for oar in oar_indices for i in oar]
	updated_D = [D_mult_max[i] for oar in oar_indices for i in oar]
	return oar_indices, T_list_mult_max, T_mult_max, updated_H, alpha_mult_max, updated_gamma, B_mult_max, updated_D, updated_C



def organ_photon_matrix(organ_name, data):
	"""This function is needed for monitoring smoothness and adjusting lambda_smoothing"""
	organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
	organ_number = organ_names.index(organ_name)
	organ_number_no_target = organ_number-1
	len_voxels = data['Aphoton'].shape[0]
	#[:-1] because we don't wabt the last isolated voxel
	organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
	#Do this in per-voxel fashion
	photon_num = data['Aphoton'].shape[1]
#     u_photon = u[:photon_num]
#     u_proton = u[photon_num:]
	organ_Aphoton = data['Aphoton'][organ_indices[organ_number]]
	return organ_Aphoton

def update_dose_volume_eta(eta_dv, eta, dv_oar_indices, data):
	"""Use eta in the appropriate places of eta_dv as initial parameters"""
	non_dv_positions = np.array(dv_oar_indices) 
	non_dv_positions[data['OAR_constraint_types'] == 'dose_volume'] *= 0 
	non_dv_positions = np.concatenate(non_dv_positions).astype('bool')
	eta_dv_output = np.array(eta_dv)
	eta_dv_output[non_dv_positions] = eta
	
	return eta_dv_output