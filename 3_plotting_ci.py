# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 07:44:50 2019

@author: Guido Riembauer
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# pick result dataset 
folder = "/media/guido/INeedMoreSpace/ESA/Harddrive/6Mh_Polimi_Aresys/code/results/"
dataset = folder + "results_noS1000_noCI1000_run3.list"
run = 3

with open(dataset, 'rb') as file:
  data = pickle.load(file)

colorlist = ["red","blue","green", "darkorange"]
outfolder = "D:/6Mh_Polimi_Aresys/code/results/figs/"

fontsize = 11

#%%###############################
# strip it all apart again
sites = [data[idx]["site"] for idx,site in enumerate(data)]

sds = []
ci_comp_corr_amps = []
ci_comp_corr_args = []

for item in data[0]["ROIs"][0].keys():
  if len(item) > 3:
    sd_temp = np.float(item.split("_")[2])
    sds.append(sd_temp)
    amp_temp = np.float(item.split("_")[4])
    #if amp_temp == 0.0:
    #  amp_temp = 0
    ci_comp_corr_amps.append(amp_temp)
    arg_temp = int(item.split("_")[6])
    ci_comp_corr_args.append(arg_temp)
    
sds = [item for item in set(sds)]
ci_comp_corr_amps = [item for item in set(ci_comp_corr_amps)]
ci_comp_corr_args = [item for item in set(ci_comp_corr_args)]
sds = sorted(sds)
ci_comp_corr_amps = sorted(ci_comp_corr_amps)
ci_comp_corr_args = sorted(ci_comp_corr_args)

for sd in sds:
  #fig = plt.figure(figsize=(len(ci_comp_corr_amps)*3.33,len(ci_comp_corr_args)*3.5))
  fig = plt.figure(figsize=(5,7))

  # get the according 3 sigma value:
  setsize = 1000000
  spawn = sd*np.random.randn(setsize)+1j*(sd*np.random.randn(setsize))
  perc_9973 = np.percentile(np.abs(spawn),99.73)
  db_val = 20*np.log10(perc_9973)
  spawn = None
  #fig.suptitle(r"$\epsilon_{1,2} = $" + str(np.round(db_val,1)) + " dB", fontsize = fontsize+4)
  
  plot_counter = 1
  #for arg in ci_comp_corr_args:
  for amp in ci_comp_corr_amps:
    
    #for amp in ci_comp_corr_amps:
    for arg in ci_comp_corr_args:
      
      #####
      # This is the subplot level
      ####
      
      ax = fig.add_subplot(len(ci_comp_corr_amps),len(ci_comp_corr_args), plot_counter)
      ax.set_title(r"|$\rho$|="+ str(amp) + "; $\\angle$$\\rho$=" + str(arg) + r"$^\circ$", fontsize = fontsize-2)
      
      # now loop through the sites and ROIs to get the statistics to be plotted
      for sidx, site in enumerate(sites):
        color = colorlist[sidx]
        rois = data[sidx]["ROIs"]
        agbs_site = []
        means_site = []
        sig2s_site = []
        sig3s_site = []
        vars_site = []
        for roi in rois:
          agb = roi["AGB"]
          agbs_site.append(agb)
          # adapt this part when the output format changes
          mean = roi["ci_sd_{}_amp_{}_arg_{}_mean".format(str(sd),str(amp),str(arg))]
          means_site.append(mean)
          sig2 = roi["ci_sd_{}_amp_{}_arg_{}_sig2".format(str(sd),str(amp),str(arg))]
          sig2s_site.append(sig2)
          sig3 = roi["ci_sd_{}_amp_{}_arg_{}_sig3".format(str(sd),str(amp),str(arg))]
          sig3s_site.append(sig3)
          var = roi["ci_sd_{}_amp_{}_arg_{}_var".format(str(sd),str(amp),str(arg))]
          vars_site.append(var)
        # plot the means
        # ax.plot(agbs_site,means_site, "+", color = color, markersize = 6)
        ax.plot(agbs_site,sig2s_site, "o", color = color, markersize = 2.5, label = site)
      ax.grid()
      ax.tick_params(axis='both', which='major', labelsize=fontsize-4)
      if plot_counter == len(ci_comp_corr_amps) * len(ci_comp_corr_args):
        ax.legend(fontsize = fontsize-4)
      ax.set_ylim([0,10])
      ax.set_xlim([100,600])
      ax.set_xlabel("AGB [t/ha]", fontsize = fontsize-4)
      ax.set_ylabel("rel. AGB error [%]", fontsize = fontsize-4)
      plot_counter = plot_counter+1
      plt.tight_layout()
  #plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.tight_layout()
  #fig.savefig(outfolder+ "run_{}_ci_sd_{}.pdf".format(run,np.round(db_val,1)), dpi = 300)

#%%############################################################################
# test plot the terms
#####################
from datetime import date
today = date.today()
path = "D:/6Mh_Polimi_Aresys/code/results/figs/paper/"
filename = path + "xt_example_sampling.png".format(today.strftime("%Y-%m-%d"))
  
from cib_functions import plot_ci_xt
plot_ci_xt(0.008,0.6,90,100000,7,filename)




