import numpy as np
import scipy.optimize
import pandas as pd
import operator
import scipy.io
import scipy
import scipy.sparse
import time
import sys
import os
from optimization_tools import *
from experiments import *

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.stats as stats


# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
def plot_beam(x_beam, y_beam, u_beam):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = x_beam#[:, None]
    Y = y_beam#[None, :]
    Z = u_beam#[None, :]
#     X, Y, Z = np.broadcast_arrays(x, y, z)
#     print(X.shape)
#     X, Y = np.meshgrid(x_beam, y_beam)
#     Z = u_beam
    # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
    surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False, cmap=cm.coolwarm)


    # Customize the z axis.
    ax.set_zlim(*stats.describe(u_beam)[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30,60)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

# #Example use:
# x_beam = data['beamlet_pos'][beamlet_indices[0]][:,0]
# y_beam = data['beamlet_pos'][beamlet_indices[0]][:,1]
# u_beam = u_mult[beamlet_indices[0]]
# plot_beam(x_beam, y_beam, u_beam)
#############################################


def dif_avg(u_beam):
    """compares top 5% to the rest"""
    u = np.sort(u_beam)[::-1]
#     print(u)
    ind = u.shape[0]//100*5
    top5 = np.mean(u[:ind])
#     bottom5 = np.mean(u[-ind:])
    mean_wo_top5 = np.mean(u[ind:])
    return top5/mean_wo_top5

#Example:[dif_avg(u_mult[beamlet_indices[i]]) for i in range(5)]
###############################################


def evaluation_function(u, N, data, organ_name, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 50):
    """Constructs DVH"""
    #Target is included here:
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
    organ_number = organ_names.index(organ_name)
    organ_number_no_target = organ_number-1
    len_voxels = data['Aphoton'].shape[0]
    #[:-1] because we don't wabt the last isolated voxel
    organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
    #Do this in per-voxel fashion
    photon_num = data['Aphoton'].shape[1]
    u_photon = u[:photon_num]
    u_proton = u[photon_num:]
    organ_Aphoton = data['Aphoton'][organ_indices[organ_number]]
    organ_Aproton = data['Aproton'][organ_indices[organ_number]]
    organ_photon_dose = organ_Aphoton.dot(u_photon) #shape of this is num_voxels
    organ_proton_dose = organ_Aproton.dot(u_proton)
    if organ_name != 'Target':
        organ_photon_BE = N[0]*Gamma[organ_number_no_target][
            0]*organ_photon_dose + N[0]*Delta[organ_number_no_target][0]*organ_photon_dose**2
        organ_proton_BE = N[1]*Gamma[organ_number_no_target][
            1]*organ_proton_dose + N[1]*Delta[organ_number_no_target][1]*organ_proton_dose**2
    if organ_name == 'Target':
        organ_photon_BE = N[0]*Alpha[0]*organ_photon_dose + N[0]*Beta[0]*organ_photon_dose**2
        organ_proton_BE = N[1]*Alpha[1]*organ_proton_dose + N[1]*Beta[1]*organ_proton_dose**2
    organ_BE = organ_photon_BE + organ_proton_BE #shape of this is num_voxels(for this OAR/organ)
    #Now we would need to compute the RHS for different d and compare each voxel to it
    #This is a TODO for tomorrow
    total_N = 45 #Standard practice - 45 fractions of Photons
#     d = np.linspace(0, max_, resolution)/total_N
#     if organ_name != 'Target':
#         lin = Gamma[organ_number_no_target][0]*total_N
#         quad = Delta[organ_number_no_target][0]*total_N    
#         BE_conventional = lin*d + quad*d**2
#     if organ_name == 'Target':
#         lin = Alpha[0]*total_N
#         quad = Beta[0]*total_N    
#         BE_conventional = lin*d + quad*d**2
    BE_levels = np.linspace(0, max_BE, resolution)
    #Now for each BE level find the fraction of voxels that are <=
    DV_fractions = []
    for BE_level in BE_levels:
        DV_fraction = np.sum(organ_BE >= BE_level)/len(organ_BE)
        DV_fractions.append(DV_fraction)
    return BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_proton_dose, organ_photon_BE, organ_proton_BE

# #Example:
# d, DV_fractions, BE_conventional, organ_BE, organ_photon_dose, organ_proton_dose = evaluation_function(u_mult, [43,1], data, 'R Femur', Alpha, Beta, Gamma, Delta, resolution = 50)
# stats.describe(organ_BE)
#############################################################


#Only photons:
def evaluation_function_photon(u, N, data, organ_name, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 50):
    """Constructs DVH"""
    #Target is included here:
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
    organ_number = organ_names.index(organ_name)
    organ_number_no_target = organ_number-1
    len_voxels = data['Aphoton'].shape[0]
    #[:-1] because we don't wabt the last isolated voxel
    organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
    #Do this in per-voxel fashion
    photon_num = data['Aphoton'].shape[1]
    u_photon = u[:photon_num]
    u_proton = u[photon_num:]
    organ_Aphoton = data['Aphoton'][organ_indices[organ_number]]
#     organ_Aproton = data['Aproton'][organ_indices[organ_number]]
    organ_photon_dose = organ_Aphoton.dot(u_photon) #shape of this is num_voxels
#     organ_proton_dose = organ_Aproton.dot(u_proton)
    if organ_name != 'Target':
        organ_photon_BE = N[0]*Gamma[organ_number_no_target][
            0]*organ_photon_dose + N[0]*Delta[organ_number_no_target][0]*organ_photon_dose**2
#         organ_proton_BE = N[1]*Gamma[organ_number_no_target][
#             1]*organ_proton_dose + N[1]*Delta[organ_number_no_target][1]*organ_proton_dose**2
    if organ_name == 'Target':
        organ_photon_BE = N[0]*Alpha[0]*organ_photon_dose + N[0]*Beta[0]*organ_photon_dose**2
#         organ_proton_BE = N[1]*Alpha[1]*organ_proton_dose + N[1]*Beta[1]*organ_proton_dose**2
    organ_BE = organ_photon_BE #+ organ_proton_BE #shape of this is num_voxels(for this OAR/organ)
    #Now we would need to compute the RHS for different d and compare each voxel to it
    #This is a TODO for tomorrow
#     total_N = 45 #Standard practice - 45 fractions of Photons
#     d = np.linspace(0, max_, resolution)/total_N
#     if organ_name != 'Target':
#         lin = Gamma[organ_number_no_target][0]*total_N
#         quad = Delta[organ_number_no_target][0]*total_N    
#         BE_conventional = lin*d + quad*d**2
#     if organ_name == 'Target':
#         lin = Alpha[0]*total_N
#         quad = Beta[0]*total_N    
#         BE_conventional = lin*d + quad*d**2
    BE_levels = np.linspace(0, max_BE, resolution)
    #Now for each BE level find the fraction of voxels that are <=
    DV_fractions = []
    for BE_level in BE_levels:
        DV_fraction = np.sum(organ_BE >= BE_level)/len(organ_BE)
        DV_fractions.append(DV_fraction)
    #Note that organ_BE and organ_photon_BE should be the same    
    print('organ_BE: ', organ_BE)
    print('organ_photon_BE: ', organ_photon_BE)
    return BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE