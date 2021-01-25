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
import gurobipy as gp
from gurobipy import GRB

data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'ProstateExample_BODY_not_reduced_with_OAR_constraints.mat'))
# data_no_body_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'ProstateExample.mat'))
data = scipy.io.loadmat(data_path)
# data_no_body =  scipy.io.loadmat(data_no_body_path)
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


# # Create a new model
# m = gp.Model("qp")

# # Create variables
# x = m.addMVar(3, ub=1.0, name="x")
# A = np.array([[1,1/2,0],
# 	          [1/2, 1, 1/2],
# 	          [0,1/2,1]])
# b = np.array([2,0,0])
# # y = m.addVar(ub=1.0, name="y")
# # z = m.addVar(ub=1.0, name="z")

# # Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
# # obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
# obj = x@A@x + b@x
# m.setObjective(obj)

# # Add constraint: x + 2 y + 3 z <= 4
# # m.addConstr(x + 2 * y + 3 * z >= 4, "c0")
# c0 = np.array([1,2,3])
# m.addConstr(c0@x >= 4, "c0")

# # Add constraint: x + y >= 1
# # m.addConstr(x + y >= 1, "c1")
# c1 = np.array([1,1,0])
# m.addConstr(c1@x >= 1, "c1")

# m.optimize()

# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

# print('Obj: %g' % obj.getValue())

# x.vType = GRB.INTEGER
# # y.vType = GRB.INTEGER
# # z.vType = GRB.INTEGER

# m.optimize()

# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

# print('Obj: %g' % obj.getValue())


# loading_dir = config_experiment+'_photon_{}_{}'.format(N_photon, 0)
# T_list_photon = utils.load_obj('T_list_photon', loading_dir)
# T_photon = utils.load_obj('T_photon', loading_dir)
# H_photon = utils.load_obj('H_photon', loading_dir)
# alpha_photon = utils.load_obj('alpha_photon', loading_dir)
# gamma_photon = utils.load_obj('gamma_photon', loading_dir)
# B_photon = utils.load_obj('B_photon', loading_dir)
# D_photon = utils.load_obj('D_photon', loading_dir)
# C_photon = utils.load_obj('C_photon', loading_dir)
def fixed_N_qcqp(N, dose_deposition_dict, constraint_dict, radbio_dict, S, alpha_smoothing):
	"""In every dict, keys are organ namesSo far, we are doing this for photons onlyS is the smoothing matrix"""
	T = dose_deposition_dict['Target']
	m = gp.Model("Fixed-N-QCQP")
	u = m.addMVar(T.shape[1], vtype=GRB.CONTINUOUS, lb=0.0, name="u")
	
	target_dose = T@u #Dose
	smoothing_term = S@u
	alpha, beta = radbio_dict['Target'] #Linear and quadratic coefficients
	obj = -N*(alpha*target_dose.sum() + beta*target_dose@target_dose) + alpha_smoothing*smoothing_term@smoothing_term
	m.setObjective(obj)

	OAR_names = list(dose_deposition_dict.keys())[1:] #Not including Target
	for oar in OAR_names:
		H = dose_deposition_dict[oar]
		oar_dose = H@u
		constraint_type, constraint_dose, constraint_N = constraint_dict[oar]
		constraint_dose = constraint_dose/constraint_N #Dose per fraction
		gamma, delta = radbio_dict[oar]
		if constraint_type == 'max_dose':
			max_constraint_BE = constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			max_constr = N*(gamma*oar_dose + delta*oar_dose**2)
			m.addConstr(max_constr <= max_constraint_BE, "{} max constraint".format(oar))
		if constraint_type_dict[oar] == 'mean_dose':
			#Sum the constr BEs across voxels:
			mean_constraint_BE = H.shape[0]*constraint_N*(gamma*constraint_dose + delta*constraint_dose**2)
			#The sum of BEs across voxels
			mean_constr = N*(gamma*oar_dose.sum() + delta*oar_dose@oar_dose)
			m.addConstr(mean_constr <= mean_constraint_BE, "{} max constraint".format(oar))

	m.optimize()
	for v in m.getVars():
		print('%s %g' % (v.varName, v.x))

	print('Obj: %g' % obj.getValue())
	utils.save_obj(u.x, 'u_photon_gurobi', '')
	print('Saved solution')
	return

dose_deposition_dict = {}
constraint_dict = {}
radbio_dict = {}

organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
len_voxels = data['Aphoton'].shape[0]
#[:-1] because we don't wabt the last isolated voxel
organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
OAR_constr_types = np.squeeze(data['OAR_constraint_types'])
OAR_constr_values = np.squeeze(data['OAR_constraint_values'])
for organ_name in organ_names:
	organ_number = organ_names.index(organ_name)
	oar_number = organ_number - 1
	dose_deposition_dict[organ_name] = data['Aphoton'][organ_indices[organ_number]]
	if organ_name == 'Target':
		radbio_dict[organ_name] = Alpha[0], Beta[0] #So far, only photons
	if organ_name != 'Target':
		constraint_type = OAR_constr_types[oar_number].strip()
		constraint_dose = OAR_constr_values[oar_number]
		constraint_N = 44
		constraint_dict[organ_name] = (constraint_type, constraint_dose, constraint_N)
		radbio_dict[organ_name] = Gamma[oar_number][0], Delta[oar_number][0]


N = 44
#Set up smoothing matrix
len_voxels = data['Aphoton'].shape[0]
beamlet_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_beamlets'])))[:-1] 
beams = [data['beamlet_pos'][i] for i in beamlet_indices]
S = utils.construct_smoothing_matrix(beams, eps = 5)
alpha_smoothing = 1e-2
print('\nDose_deposition_dict:', dose_deposition_dict)
print('\nConstraint dict:', constraint_dict)
print('\nradbio_dict:', radbio_dict)
print('\nN:', N)
print('\nS shape:', S.shape)

print('\n Running optimization...')
fixed_N_qcqp(N, dose_deposition_dict, constraint_dict, radbio_dict, S, alpha_smoothing)






