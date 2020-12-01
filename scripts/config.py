import numpy as np




configurations = {}




#Experiment 1: Two modality, dosimetric difference
def experiment_1():
	"""Config for Experiment 1: Two modality, dosimetric difference"""
	configurations = {}
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

	configurations['Alpha'] = Alpha
	configurations['Beta'] = Beta
	configurations['Gamma'] = Gamma
	configurations['Delta'] = Delta
	configurations['modality_names'] = modality_names
	return configurations


configurations['Experiment_1': experiment_1()]