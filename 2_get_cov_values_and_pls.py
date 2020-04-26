# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:49:31 2019
best results for Lope with T2-0,Ch4-0,Ch4-1,Ch4-2 stack (a = 5.94), when master = Ch4-2,
use the earlier sigma-version instead
best results for Mabounie with CMA2-3 stack (a = 9.77)
best results for paracou: using gamma0; a = 11.84, when using 5 as a master instead of 0, emphasizing 50m 
@author: Guido Riembauer
"""
#

#import gdal
#import matplotlib.pyplot as plt
#import pdb
from cib_functions import get_stats, powerLaw, makeS2, plot_ci_xt,agb_error_stats
from maap_processing import fit_pl
import numpy as np
import pickle
#import multiprocessing as mp
#import concurrent.futures

# define sites
sites = ["Mabounie","Paracou"] # ["Lope","Mabounie","Paracou"]
a_pl = [9.77,11.53] # 8.07,9.77,11.53 --> new one for Lope: 14.99
b_pl = [-35.31,-38.97] # -28.61,-35.31,-38.97 -> new one for Lope: -47.24

# define error terms and simulation parameters
noS = 1000            # number of pixels per simulated scene
noCI = 10000           # number of simulations per ROI, used for every simulation run
tol = 3.0               # tolerancein % of deviation of the simulated S matrix
NESZ = -27.             # Noise Equivalent Sigma Zero
XT_sd = 0.01          # standard deviation of the complex XT terms; this equals a magnitude of roughly -30 db as expected, can be tested using the plot_ci_xt function
XT_amp_comp_corr = 0.8  # magnitude of the XT complex correlation coefficient; does not have a big influence
XT_arg_comp_corr = 0    # phase difference of the XT complex correlation coefficient; does not have a big influence

CI_sds = [0.009,0.012,0.014]      # standard deviation of the complex ci terms, they translate to roughly -30; -27.5; -26 dB ad 3 sigma confidence
CI_amp_comp_corrs = [0.0,0.3,0.8,0.99]    # magnitude of the XT complex correlation coefficient
CI_arg_comp_corrs = [0,90,180]     # phase difference of the XT complex correlation coefficient

PBE=20

result = [{"site":sites[idx]} for idx,item in enumerate(np.arange(len(sites)))]

#%%############################################################################
# get all the required data
###############################################################################
for site_idx,site in enumerate(sites):

  ROI_path = "/home/guido/paperstuff/L2/Data/Insitu/agb_attached_labriere/{}/1ha/".format(site)
  if site == "Lope":
    dataset = "afrisar_dlr_T2-0"
  elif site == "Mabounie":
    dataset = "afrisar_dlr_CMA2-3"
  elif site == "Paracou":
    dataset = "tropisar_C402"
    ROI_path = "/home/guido/paperstuff/L2/Data/Insitu/agb_attached_labriere/Paracou/6.25ha/"
  
  folder = "/media/guido/INeedMoreSpace/ESA/Harddrive/6Mh_Polimi_Aresys/code/results/"
  
  hh_file = folder + dataset + "_HH_gc.tiff"
  #hv_file = folder + dataset + "_HV_gc.tiff"
  vh_file = folder + dataset + "_VH_gc.tiff"
  vv_file = folder + dataset + "_VV_gc.tiff"
  
  cov_real_file = folder + dataset + "_gc_HH_VV_cov_real.tiff"
  cov_imag_file = folder + dataset + "_gc_HH_VV_cov_imag.tiff"
  
  # get the pl parameters and refagb 
  #a_old,b_old,refagb,rest = fit_pl(vh_file,ROI_path)
  a,b, refagb, backscatter_log,listlinregress = fit_pl(vh_file,ROI_path)
  # overwrite the a and b parameters as defined before (for Mabounie and Paracou they come from the self-notched data, for Lope from an older Aresys version)
  a = a_pl[site_idx]
  b = b_pl[site_idx]
  result[site_idx]["a"] = a
  result[site_idx]["b"] = b
  print("Processing site {} with pl parameters a = {}, b = {}".format(site,a,b))
  
  # get the backscatter and covariance
  hh = get_stats(hh_file,ROI_path)
  vh = get_stats(vh_file,ROI_path)
  vv = get_stats(vv_file,ROI_path)
  cov_real = get_stats(cov_real_file,ROI_path)
  cov_imag = get_stats(cov_imag_file,ROI_path)
  
  # get the theta and gamma as required
  covariance = [re+ 1j*cov_imag[i] for i, re in enumerate(cov_real)]
  # gamma = normalised corrcoef = magnitude of covariance/sqrt(hh*vv)
  # thetas = phase of covariance
  thetas = [np.angle(cov) for cov in covariance]
  gammas = [np.absolute(cov)/np.sqrt(hh[idx]*vv[idx]) for idx,cov in enumerate(covariance)]
  
  # the power law 
  A_pl = 10**(-b/a)
  p_pl = 10/a
  #%%##########################################################################
  # Loop through the ROIs and the scenarios
  #############################################################################
  
  roilist = [{"AGB":refagb[ragb_idx]} for ragb_idx,agb in enumerate(np.arange(len(refagb)))] # a list of dictionaries to come
  
  for ragb_idx, agb in enumerate(refagb):
    print("Processing ROI {} of {} from site {}".format(ragb_idx+1,len(refagb),site))
    
    sigma_hh = 10*np.log10(hh[ragb_idx])
    sigma_hv = 10*np.log10(vh[ragb_idx])
    sigma_vv = 10*np.log10(vv[ragb_idx])
    theta = thetas[ragb_idx]
    gamma = gammas[ragb_idx]
    
    # generate the scene
    S = makeS2(tol,noS,sigma_hh,sigma_hv,sigma_vv,theta,gamma)
    Sbiomass = powerLaw(np.sum(abs(S[1])**2)/noS,A_pl,p_pl)
    print("Original AGB: {} t/ha; expected AGB based on pl: {}".format(agb,Sbiomass))
    
    # loop through all scenarios
    counter = 1
    for ci_sd_idx, ci_sd in enumerate(CI_sds):
      
      for ci_amp_comp_corr_idx,ci_amp_comp_corr in enumerate(CI_amp_comp_corrs):
        
        for ci_arg_comp_corr_idx, ci_arg_comp_corr in enumerate(CI_arg_comp_corrs):
          
          print("Processing Scenario {} of {} for ROI {} at site {}".format(counter,len(CI_sds)*len(CI_amp_comp_corrs)*len(CI_arg_comp_corrs),ragb_idx+1,site))
          # run the simulation
          mean, sig2, sig3, variance,prob = agb_error_stats(noS,S,Sbiomass,noCI,A_pl,p_pl, NESZ,ci_sd,
                                                       ci_amp_comp_corr,ci_arg_comp_corr,XT_sd,
                                                       XT_amp_comp_corr, XT_arg_comp_corr,PBE)
          
          roilist[ragb_idx]["ci_sd_{}_amp_{}_arg_{}_mean".format(ci_sd,ci_amp_comp_corr,ci_arg_comp_corr)] = mean
          roilist[ragb_idx]["ci_sd_{}_amp_{}_arg_{}_sig2".format(ci_sd,ci_amp_comp_corr,ci_arg_comp_corr)] = sig2
          roilist[ragb_idx]["ci_sd_{}_amp_{}_arg_{}_sig3".format(ci_sd,ci_amp_comp_corr,ci_arg_comp_corr)] = sig3
          roilist[ragb_idx]["ci_sd_{}_amp_{}_arg_{}_var".format(ci_sd,ci_amp_comp_corr,ci_arg_comp_corr)] = variance
          roilist[ragb_idx]["ci_sd_{}_amp_{}_arg_{}_probpbe".format(ci_sd,ci_amp_comp_corr,ci_arg_comp_corr)] = prob
          counter = counter+1
  result[site_idx]["ROIs"] = roilist
  

#%%############################################################################
#save the result to disk
with open(folder+"results_noS1000_noCI1000_newtest.list", 'wb') as config_dictionary_file:
# 

  pickle.dump(result, config_dictionary_file)


#%%############################################################################
# test the error terms
###############################################################################
  
#  sd = 0.014
#  amp_comp_corr = 0.6
#  arg_comp_corr = 90
#  noS = 100000
#  
#  plot_ci_xt(sd,amp_comp_corr,arg_comp_corr,noS)






