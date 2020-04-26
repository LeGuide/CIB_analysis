# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:32:24 2020

@author: Guido Riembauer
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from cib_functions import get_stats, powerLaw, makeS2, plot_ci_xt,agb_error_stats
from maap_processing import fit_pl
import pickle
from datetime import date

# take one Lope dataset as example
site = "Lope"
roi_idx = 4
a = 8.07
b = -28.61
tol = 2 # for makeS
noS = 1000 # for makeS
noCI = 10000 # for looping
NESZ = -27.

# one tiny, one realistic, for both ci and xt
sd_test = 0.012 
#amp_comp_corrs_test = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
amp_comp_corrs_test = [0.1,0.9]
argument_compcorr_test = np.arange(0,190,10)
sd_dummy = 0.00001
amp_comp_corrs_dummy = 0.5
argument_compcorr_dummy = 0


###############################################################################
#######
# read and transform all the required data
ROI_path = "C:/3_BIOMASS/L2/Data/Insitu/agb_attached_labriere/{}/1ha/".format(site)
if site == "Lope":
  dataset = "afrisar_dlr_T2-0"

folder = "D:/6Mh_Polimi_Aresys/code/results/"
  
hh_file = folder + dataset + "_HH_gc.tiff"
#hv_file = folder + dataset + "_HV_gc.tiff"
vh_file = folder + dataset + "_VH_gc.tiff"
vv_file = folder + dataset + "_VV_gc.tiff"

a_old,b_old,refagb, rest = fit_pl(vh_file,ROI_path)
cov_real_file = folder + dataset + "_gc_HH_VV_cov_real.tiff"
cov_imag_file = folder + dataset + "_gc_HH_VV_cov_imag.tiff"
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


hh = hh[roi_idx]
vh = vh[roi_idx]
vv = vv[roi_idx]
theta = thetas[roi_idx]
gamma = gammas[roi_idx]
agb = refagb[roi_idx]
sigma_hh = 10*np.log10(hh)
sigma_hv = 10*np.log10(vh)
sigma_vv = 10*np.log10(vv)


#######
# create an S matrix
S = makeS2(tol,noS,sigma_hh,sigma_hv,sigma_vv,theta,gamma)
Sbiomass = powerLaw(np.sum(abs(S[1])**2)/noS,A_pl,p_pl)
print("Original AGB: {} t/ha; expected AGB based on pl: {}".format(agb,Sbiomass))

######
# loop through scenarios

# scenario 1: varying xt, ci is zero
means_xt = []
sig2s_xt = []
sig3s_xt = []
variances_xt = []

means_ci = []
sig2s_ci = []
sig3s_ci = []
variances_ci = []

run = 0
for amp_compcorr in amp_comp_corrs_test:
    for arg_compcorr in argument_compcorr_test :
      mean_xt, sig2_xt, sig3_xt, variance_xt = agb_error_stats(noS,S,Sbiomass,noCI,A_pl,p_pl, NESZ,sd_dummy,
                                                       amp_comp_corrs_dummy,argument_compcorr_dummy,sd_test,
                                                       amp_compcorr, arg_compcorr)
      print("Done inbetween")
      mean_ci, sig2_ci, sig3_ci, variance_ci = agb_error_stats(noS,S,Sbiomass,noCI,A_pl,p_pl, NESZ,sd_test,
                                                       amp_compcorr,arg_compcorr,sd_dummy,
                                                       amp_comp_corrs_dummy, argument_compcorr_dummy)
      means_xt.append(mean_xt)
      sig2s_xt.append(sig2_xt)
      sig3s_xt.append(sig3_xt)
      variances_xt.append(variance_xt)
      means_ci.append(mean_ci)
      sig2s_ci.append(sig2_ci)
      sig3s_ci.append(sig3_ci)
      variances_ci.append(variance_ci)
      run = run+1
      print("Run {} of {} done".format(run, len(amp_comp_corrs_test)*len(argument_compcorr_test)))

#%%
######
# save the result to disk, first create a proper dictionary
resultdict = {"sim_properties":{"a":a, "b":b, "roi_idx":roi_idx, "agb":agb, "noS":noS,"noCI":noCI, "NESZ":NESZ, 
                                "sd_test":sd_test, "amp_compcorrs_test": amp_comp_corrs_test, "arg_compcorrs_test": argument_compcorr_test,
                                "sd_dummy":sd_dummy, "amp_compcorr_dummy":amp_comp_corrs_dummy, "arg_compcorr_dummy": argument_compcorr_dummy},
              "ci_varies":{"means":means_ci,"sig2s":sig2s_ci, "sig3s":sig3s_ci, "variances": variances_ci},
              "xt_varies":{"means":means_xt,"sig2s":sig2s_xt, "sig3s":sig3s_xt, "variances": variances_xt}}

today = date.today()
filename = "D:/6Mh_Polimi_Aresys/code/results/" + "cib_xt_influence_analysis_" + today.strftime("%Y-%m-%d") + ".pickle"
  
with open(filename, 'wb') as outfile:
# 

  pickle.dump(resultdict, outfile)
  
#%%
# plot stuff
  
with open(filename, 'rb') as file:
  data = pickle.load(file)
  
fig = plt.figure(figsize=(5,2.5))
fontsize = 12

for i in [1,2]:
  ax = fig.add_subplot(1,2,i)
  x_axis = data["sim_properties"]["arg_compcorrs_test"]
  if i == 1:
    ax.set_title("(a)", fontsize = fontsize)
    ax.plot(x_axis, data["ci_varies"]["means"][0:19],linestyle = "dashed", color = "red")
    ax.plot(x_axis, data["ci_varies"]["variances"][0:19],linestyle = "solid", color = "red")
    ax.plot(x_axis, data["ci_varies"]["means"][19:],linestyle = "dashed", color = "blue")
    ax.plot(x_axis, data["ci_varies"]["variances"][19:],linestyle = "solid", color = "blue")
  elif i == 2:
    ax.set_title("(b)", fontsize = fontsize)
    ax.plot(x_axis, data["xt_varies"]["means"][0:19],linestyle = "dashed", color = "red", label = r"mean ($|\rho| = 0.1$)")
    ax.plot(x_axis, data["xt_varies"]["variances"][0:19],linestyle = "solid", color = "red", label = r"var ($|\rho| = 0.1$)")
    ax.plot(x_axis, data["xt_varies"]["means"][19:],linestyle = "dashed", color = "blue", label = r"mean ($|\rho| = 0.9$)")
    ax.plot(x_axis, data["xt_varies"]["variances"][19:],linestyle = "solid", color = "blue", label = r"var ($|\rho| = 0.9$)")
    ax.legend(fontsize = fontsize-5)
    
  ax.grid()
  ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
  ax.set_xticks([0,45,90,135,180])
  ax.set_xlim((0,180))
  ax.set_ylim((-1,9))
  ax.set_xlabel(r"$\angle\rho [^\circ]$")
  ax.set_ylabel("rel. AGB error [%]")
  
plt.tight_layout()
path = "D:/6Mh_Polimi_Aresys/code/results/figs/paper/"
outfigname = path + "cib_xt_influence_{}.pdf".format(today.strftime("%Y-%m-%d"))
#fig.savefig(outfigname, dpi = 300)


