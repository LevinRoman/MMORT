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


# Create a new model
m = gp.Model("qp")

# Create variables
x = m.addMVar(3, ub=1.0, name="x")
A = np.array([[1,1/2,0],
	          [1/2, 1, 1/2],
	          [0,1/2,1]])
b = np.array([2,0,0])
# y = m.addVar(ub=1.0, name="y")
# z = m.addVar(ub=1.0, name="z")

# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
# obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
obj = x.T@A@x + b@x
m.setObjective(obj)

# Add constraint: x + 2 y + 3 z <= 4
# m.addConstr(x + 2 * y + 3 * z >= 4, "c0")
c0 = np.array([1,2,3])
m.addConstr(c0@x >= 4, "c0")

# Add constraint: x + y >= 1
# m.addConstr(x + y >= 1, "c1")
c1 = np.array([1,1,0])
m.addConstr(c1@x >= 1, "c1")

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())

x.vType = GRB.INTEGER
y.vType = GRB.INTEGER
z.vType = GRB.INTEGER

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())