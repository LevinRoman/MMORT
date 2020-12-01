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
#Note that none of the functions are able to handle mean-dose constraint

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

def evaluation_mult_plot_BE(path, ax_BE, ax_dose, u, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500):
    #Note that mean dose is not supported
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
#     evaluation_results = []
    list_of_BEs_and_names = []
    list_of_doses_and_names = []
    for organ in organ_names:
        BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_proton_dose, organ_photon_BE, organ_proton_BE = evaluation_function(u, N, data, organ, Alpha, Beta, Gamma, Delta, max_BE, resolution)
        list_of_BEs_and_names.append([organ, organ_BE])
        
        d, d_fractions = dose_dvh(max_dose, dose_resolution, N[0]*organ_photon_dose + N[1]*organ_proton_dose)
        list_of_doses_and_names.append([organ, (N[0]*organ_photon_dose + N[1]*organ_proton_dose)/(np.sum(N))])
        
        organ_constraint_dose, organ_constraint_BE, organ_constraint_fraction = organ_constraint(N, data, organ, Alpha, Beta, Gamma, Delta)
        
        
        ax_BE.plot(BE_levels, DV_fractions, label = organ)
        ax_dose.plot(d, d_fractions, label = organ)
        if organ != 'Target':
            if organ_constraint_fraction < 1.0: #if dvc
                ls = '--'
                label = 'DVC'
                ax_BE.hlines(organ_constraint_fraction, 0, organ_constraint_BE, ls = ls, colors = 'b')
                ax_dose.hlines(organ_constraint_fraction, 0, organ_constraint_dose, ls = ls, colors = 'b')
            else:
                ls = '-.'
                label = 'MaxDose'
            ax_BE.vlines(organ_constraint_BE, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            ax_dose.vlines(organ_constraint_dose, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            
            
    ax_BE.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_BE.set_xlim(xmin=-1, xmax = max_BE)  
    ax_BE.legend()
    ax_BE.set_ylabel('Fraction')
    ax_BE.set_xlabel('BE')
    ax_BE.set_title('BE DVH: {} Photons {} Protons'.format(N[0], N[1]))
    
    
    ax_dose.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_dose.set_xlim(xmin=-1, xmax = max_dose)
    ax_dose.legend()
    ax_dose.set_ylabel('Fraction')
    ax_dose.set_xlabel('Dose (Gy)')
    ax_dose.set_title('Dose DVH: {} Photons {} Protons'.format(N[0], N[1]))
#     evaluation_results.append(BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE)
    min_max_mean_df(list_of_BEs_and_names, dose_type = '_mult_BE').to_csv(os.path.join(path, 'mult_BE_df.csv'))
    min_max_mean_df(list_of_doses_and_names, dose_type = '_mult_Dose').to_csv(os.path.join(path, 'mult_dose_df.csv'))
    return

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


def dose_dvh(max_BE, resolution, organ_BE):
    BE_levels = np.linspace(0, max_BE, resolution)
    #Now for each BE level find the fraction of voxels that are <=
    DV_fractions = []
    for BE_level in BE_levels:
        DV_fraction = np.sum(organ_BE >= BE_level)/len(organ_BE)
        DV_fractions.append(DV_fraction)
    return BE_levels, DV_fractions

def min_max_mean_df(list_of_doses_and_names, dose_type = '_BE'):
    """list_of_doses_and_names should be a list of tuples"""
    df_list = [[i[0], np.min(i[1]), np.max(i[1]), np.mean(i[1])] for i in list_of_doses_and_names]
    df_cols = ['Organ', 'min'+dose_type, 'max'+dose_type, 'mean'+dose_type]
    df = pd.DataFrame(df_list, columns=df_cols)
    return df

def evaluation_photon_plot_BE(path, ax_BE, ax_dose, u, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500):
    #Note that mean dose is not supported
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
#     evaluation_results = []
    list_of_BEs_and_names = []
    list_of_doses_and_names = []
    for organ in organ_names:
        BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE = evaluation_function_photon(u, N, data, organ, Alpha, Beta, Gamma, Delta, max_BE, resolution)
        list_of_BEs_and_names.append([organ, organ_BE])
        
        d, d_fractions = dose_dvh(max_dose, dose_resolution, N[0]*organ_photon_dose)
        list_of_doses_and_names.append([organ, organ_photon_dose])
        
        organ_constraint_dose, organ_constraint_BE, organ_constraint_fraction = organ_constraint(N, data, organ, Alpha, Beta, Gamma, Delta)
        
        
        ax_BE.plot(BE_levels, DV_fractions, label = organ)
        ax_dose.plot(d, d_fractions, label = organ)
        if organ != 'Target':
            if organ_constraint_fraction < 1.0: #if dvc
                ls = '--'
                label = 'DVC'
                ax_BE.hlines(organ_constraint_fraction, 0, organ_constraint_BE, ls = ls, colors = 'b')
                ax_dose.hlines(organ_constraint_fraction, 0, organ_constraint_dose, ls = ls, colors = 'b')
            else:
                ls = '-.'
                label = 'MaxDose'
            ax_BE.vlines(organ_constraint_BE, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            ax_dose.vlines(organ_constraint_dose, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            
            
    ax_BE.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_BE.set_xlim(xmin=-1, xmax = max_BE)  
    ax_BE.legend()
    ax_BE.set_ylabel('Fraction')
    ax_BE.set_xlabel('BE')
    ax_BE.set_title('BE DVH: {} Photons'.format(N[0]))
    
    
    ax_dose.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_dose.set_xlim(xmin=-1, xmax = max_dose)
    ax_dose.legend()
    ax_dose.set_ylabel('Fraction')
    ax_dose.set_xlabel('Dose (Gy)')
    ax_dose.set_title('Dose DVH: {} Photons'.format(N[0]))
#     evaluation_results.append(BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE)
    min_max_mean_df(list_of_BEs_and_names, dose_type = '_Photon_BE').to_csv(os.path.join(path, 'Photon_BE_df.csv'))
    min_max_mean_df(list_of_doses_and_names, dose_type = '_Photon_Dose').to_csv(os.path.join(path, 'Photon_dose_df.csv'))
    return
    
def organ_constraint(N, data, organ_name, Alpha, Beta, Gamma, Delta):
    total_constraint_N = 45 #conventional is 45 fractions of photons
    organ_constraint_BE = None
    organ_constraint_dose = None
    organ_constraint_fraction = None
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
    organ_number = organ_names.index(organ_name)
    organ_number_no_target = organ_number-1
    if organ_name != 'Target':
        organ_constraint_dose = np.squeeze(data['OAR_constraint_values'])[organ_number_no_target]
        organ_constraint_dose_per_fraction = np.squeeze(data['OAR_constraint_values'])[organ_number_no_target]/total_constraint_N
        organ_constraint_BE = total_constraint_N*Gamma[organ_number_no_target][
            0]*organ_constraint_dose_per_fraction + total_constraint_N*Delta[organ_number_no_target][0]*organ_constraint_dose_per_fraction**2
        #Create this in the data dictionary with 0.5 for dvc organs and 1.0 for max dose constrained
        organ_constraint_fraction = data['OAR_constraint_fraction'][organ_number_no_target]
    #Return total constraint dose
    return organ_constraint_dose, organ_constraint_BE, organ_constraint_fraction

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
#     print('organ_BE: ', organ_BE)
#     print('organ_photon_BE: ', organ_photon_BE)
    return BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE


def evaluation_proton_plot_BE(path, ax_BE, ax_dose, u, N, data, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 500, max_dose = 45*5.0, dose_resolution = 500):
    #Note that mean dose is not supported
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
#     evaluation_results = []
    list_of_BEs_and_names = []
    list_of_doses_and_names = []
    for organ in organ_names: 
        BE_levels, DV_fractions, organ_BE, organ_proton_dose, organ_proton_BE = evaluation_function_proton(u, N, data, organ, Alpha, Beta, Gamma, Delta, max_BE, resolution)
        list_of_BEs_and_names.append([organ, organ_BE])
        
        d, d_fractions = dose_dvh(max_dose, dose_resolution, N[1]*organ_proton_dose)
        list_of_doses_and_names.append([organ, organ_proton_dose])
        
        organ_constraint_dose, organ_constraint_BE, organ_constraint_fraction = organ_constraint(N, data, organ, Alpha, Beta, Gamma, Delta)
        
        
        ax_BE.plot(BE_levels, DV_fractions, label = organ)
        ax_dose.plot(d, d_fractions, label = organ)
        if organ != 'Target':
            if organ_constraint_fraction < 1.0: #if dvc
                ls = '--'
                label = 'DVC'
                ax_BE.hlines(organ_constraint_fraction, 0, organ_constraint_BE, ls = ls, colors = 'b')
                ax_dose.hlines(organ_constraint_fraction, 0, organ_constraint_dose, ls = ls, colors = 'b')
            else:
                ls = '-.'
                label = 'MaxDose'
            ax_BE.vlines(organ_constraint_BE, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            ax_dose.vlines(organ_constraint_dose, 0, organ_constraint_fraction, ls = ls, colors = 'b', label = label)
            
            
    ax_BE.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_BE.set_xlim(xmin=-1, xmax = max_BE)  
    ax_BE.legend()
    ax_BE.set_ylabel('Fraction')
    ax_BE.set_xlabel('BE')
    ax_BE.set_title('BE DVH: {} Protons'.format(N[1]))
    
    
    ax_dose.set_ylim(ymin=-0.01, ymax = 1.01)
    ax_dose.set_xlim(xmin=-1, xmax = max_dose)
    ax_dose.legend()
    ax_dose.set_ylabel('Fraction')
    ax_dose.set_xlabel('Dose (Gy)')
    ax_dose.set_title('Dose DVH: {} Protons'.format(N[1]))
#     evaluation_results.append(BE_levels, DV_fractions, organ_BE, organ_photon_dose, organ_photon_BE)
    min_max_mean_df(list_of_BEs_and_names, dose_type = '_Proton_BE').to_csv(os.path.join(path, 'Proton_BE_df.csv'))
    min_max_mean_df(list_of_doses_and_names, dose_type = '_Proton_Dose').to_csv(os.path.join(path, 'Proton_dose_df.csv'))
    return


#Only protons:
def evaluation_function_proton(u, N, data, organ_name, Alpha, Beta, Gamma, Delta, max_BE = 200, resolution = 50):
    """Constructs DVH"""
    #Target is included here:
    organ_names = [str(i[0]) for i in np.squeeze(data['Organ'])]
    organ_number = organ_names.index(organ_name)
    organ_number_no_target = organ_number-1
    len_voxels = data['Aphoton'].shape[0]
    #[:-1] because we don't want the last isolated voxel
    organ_indices = np.split(np.arange(len_voxels), np.cumsum(np.squeeze(data['num_voxels'])))[:-1]
    #Do this in per-voxel fashion
    proton_num = data['Aproton'].shape[1]
    u_photon = u[:-proton_num]
    u_proton = u[-proton_num:]
#     organ_Aphoton = data['Aphoton'][organ_indices[organ_number]]
    organ_Aproton = data['Aproton'][organ_indices[organ_number]]
#     organ_photon_dose = organ_Aphoton.dot(u_photon) #shape of this is num_voxels
    organ_proton_dose = organ_Aproton.dot(u_proton)
    if organ_name != 'Target':
#         organ_photon_BE = N[0]*Gamma[organ_number_no_target][
#             0]*organ_photon_dose + N[0]*Delta[organ_number_no_target][0]*organ_photon_dose**2
        organ_proton_BE = N[1]*Gamma[organ_number_no_target][
            1]*organ_proton_dose + N[1]*Delta[organ_number_no_target][1]*organ_proton_dose**2
    if organ_name == 'Target':
#         organ_photon_BE = N[0]*Alpha[0]*organ_photon_dose + N[0]*Beta[0]*organ_photon_dose**2
        organ_proton_BE = N[1]*Alpha[1]*organ_proton_dose + N[1]*Beta[1]*organ_proton_dose**2
    organ_BE = organ_proton_BE #shape of this is num_voxels(for this OAR/organ)
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
    print('organ_photon_BE: ', organ_proton_BE)
    return BE_levels, DV_fractions, organ_BE, organ_proton_dose, organ_proton_BE