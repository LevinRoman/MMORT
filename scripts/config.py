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

#Experiment 2: IMRT objective using our code
def experiment_2(N1, d):
	"""Config for Experiment 2: This experiment should
	 yield Tu-d as the objective where d is the target uniform dose per fraction"""
	configurations = {}
	Alpha = np.array([2*d/N1, 0.0])
	Beta = np.array([-1/N1, 0.0])
	Gamma = np.array([np.array([0.35, 0.0]),
	                  np.array([0.35, 0.0]),
	                  np.array([0.35, 0.0]),
	                  np.array([0.35, 0.0]),
	                  np.array([0.35, 0.0])               
	                 ])
	Delta = np.array([np.array([0.07, 0.0]),
	                  np.array([0.07, 0.0]),
	                  np.array([0.175, 0.0]),
	                  np.array([0.175, 0.0]),
	                  np.array([0.175, 0.0])                
	                 ])
	modality_names = np.array(['Aphoton', 'Aproton'])

	configurations['Alpha'] = Alpha
	configurations['Beta'] = Beta
	configurations['Gamma'] = Gamma
	configurations['Delta'] = Delta
	configurations['modality_names'] = modality_names

	return configurations

configurations['Experiment_1'] = experiment_1()
configurations['Experiment_2'] = experiment_2(44.0, 81.0/44.0)

configurations['Experiment_3'] = experiment_2(44.0, 85.0/44.0)
configurations['Experiment_4'] = experiment_2(44.0, 60.0/44.0)
configurations['Experiment_5'] = experiment_2(44.0, 100.0/44.0)

#Try breaking the uniform dose:
configurations['Experiment_6'] = experiment_2(44.0, 150.0/44.0)