# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:44:01 2019

@author: Guido Riembauer
"""

import numpy as np
import csv
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy import linalg as linalg
from scipy import stats
import rasterstats
import gdal
import rasterio
import geopandas as gpd
import os 
import pdb


def get_stats(raster, ROI_path):
  raster_crs = rasterio.open(raster).crs.data
  backscatter = []
  for item in os.listdir(ROI_path):
    if item.endswith('.shp'):
        filepath = ROI_path+item
        
        # reproject to raster's crs
        shapefile = gpd.read_file(filepath)
        shapefile = shapefile.to_crs(raster_crs)
        zstats = rasterstats.zonal_stats(shapefile,raster, nodata = -999)
        mean_bs = zstats[0]['mean']
        
        
        if shapefile.agb_loc[0] != 0.0:
          if mean_bs != None:
            if mean_bs != 0.0:
              backscatter.append(mean_bs)
  return backscatter


# New function to define XT or CIB on a more general level based on a complex SD valid for both real and imag. 
def errorterm(term,complexsd,comp_corr_abs,comp_corr_phase,noCI):
    if term == "ci":
        cov = np.array([[complexsd**2,comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase))],[np.conjugate(comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase))),complexsd**2]])
        chol = linalg.cholesky(cov)
        randoms = np.random.randn(noCI,2) + 1j * np.random.randn(noCI,2) 
        samples = np.conjugate(randoms@chol).T
    elif term == "xt":
        cov = np.array([[complexsd**2,0,comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase)),0],
                        [0,complexsd**2,0,comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase))],
                        [np.conjugate(comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase))),0,complexsd**2,0],
                        [0,np.conjugate(comp_corr_abs*complexsd**2*np.exp(1j*np.radians(comp_corr_phase))),0,complexsd**2]])
        chol = linalg.cholesky(cov)
        randoms = np.random.randn(noCI,4) + 1j * np.random.randn(noCI,4) 
        samples = np.conjugate(randoms@chol).T
    else:
        print("Please indicate 'ci' or 'xt' as first argument")
    return samples
  
# Noise Term
def makeN(NESZ,noS,n):
    N = (np.random.randn(noS,n) + np.random.randn(noS,n)*1j)/2**0.5*10**(NESZ/20)
    # can be tested using
    # 10*np.log10(np.mean(abs(N[x])**2)) 
    # the **2 since N is amplitude and not intensity while NESZ is intensity (squared amplitude)
    return N

# The power law
def powerLaw(sigma_hv,A,p):
    return(A*sigma_hv**p)
    # to convert from/to a and b as in Schlund: 
    # A = 10**(-b/a)
    # p = 10/a

# build a covariance matrix using the input values
def makeSCov(shhMod2,shvMod2,svvMod2,theta,gamma):
    sCov = np.array([[shhMod2, 0, (shhMod2*svvMod2)**0.5*gamma*np.exp(1j*-theta)],[0, shvMod2,0],[(shhMod2*svvMod2)**0.5*gamma*np.exp(1j*theta),0,svvMod2]])
    return(sCov)

# generate S-matrix
def makeS(tol,noS,sigma_hh,sigma_hv,sigma_vv,theta,gamma):
    shhMod2 = 10**(sigma_hh/10)
    shvMod2 = 10**(sigma_hv/10)
    svvMod2 = 10**(sigma_vv/10)
    i = 0
    iterations = 0
    while i == 0:
        iterations = iterations+1
        sCov = makeSCov(shhMod2,shvMod2,svvMod2,theta,gamma)
        sCovChol = linalg.cholesky(sCov)
        Srandom = np.random.randn(noS,3)*np.exp(1j*np.random.rand(noS,3)*2*np.pi)
        S = np.conjugate(Srandom@sCovChol).T
        S = np.stack((S[0],S[1],S[1],S[2]))
        Shh = S[0]
        Shv = S[1]
        Svh = S[2]
        Svv = S[3]
        cc = np.corrcoef(Shh,Svv)
        if abs(shhMod2-np.sum(Shh*np.conjugate(Shh))/len(Shh))/shhMod2*100 < tol and \
        abs(shvMod2-np.sum(Shv*np.conjugate(Shv))/len(Shv))/shvMod2*100 < tol and \
        abs(svvMod2-np.sum(Svv*np.conjugate(Svv))/len(Svv))/svvMod2*100 < tol and \
        abs(theta-np.angle(Shh@Svv.T))/abs(theta)*100 < tol and \
        abs(gamma-abs(cc[0,1]))/gamma*100 < tol:
            i = 1
            
    print("S matrix within tolerance created after " + str(iterations) + " iterations")
    print("sigma_hh = " + str(np.round(10*np.log10(np.sum(Shh*np.conjugate(Shh))/len(Shh)),4)))
    print("sigma_hv = " + str(np.round(10*np.log10(np.sum(Shv*np.conjugate(Shv))/len(Shv)),4)))
    print("sigma_vv = " + str(np.round(10*np.log10(np.sum(Svv*np.conjugate(Svv))/len(Svv)),4)))
    print("angle = " + str(np.round(np.angle(Shh@Svv.T)*180/np.pi,4)))
    print("gamma = " + str(np.round(abs(cc[0,1]),4)))
    
    return S        

# another makeS, but with short print output, for many runs
# generate S-matrix
def makeS2(tol,noS,sigma_hh,sigma_hv,sigma_vv,theta,gamma):
    shhMod2 = 10**(sigma_hh/10)
    shvMod2 = 10**(sigma_hv/10)
    svvMod2 = 10**(sigma_vv/10)
    i = 0
    iterations = 0
    while i == 0:
        iterations = iterations+1
        sCov = makeSCov(shhMod2,shvMod2,svvMod2,theta,gamma)
        sCovChol = linalg.cholesky(sCov)
        Srandom = np.random.randn(noS,3)*np.exp(1j*np.random.rand(noS,3)*2*np.pi)
        S = np.conjugate(Srandom@sCovChol).T
        S = np.stack((S[0],S[1],S[1],S[2]))
        Shh = S[0]
        Shv = S[1]
        Svh = S[2]
        Svv = S[3]
        cc = np.corrcoef(Shh,Svv)
        if abs(shhMod2-np.sum(Shh*np.conjugate(Shh))/len(Shh))/shhMod2*100 < tol and \
        abs(shvMod2-np.sum(Shv*np.conjugate(Shv))/len(Shv))/shvMod2*100 < tol and \
        abs(svvMod2-np.sum(Svv*np.conjugate(Svv))/len(Svv))/svvMod2*100 < tol and \
        abs(theta-np.angle(Shh@Svv.T))/abs(theta)*100 < tol and \
        abs(gamma-abs(cc[0,1]))/gamma*100 < tol:
            i = 1
            
    print("S matrix within tolerance created after " + str(iterations) + " iterations")
    return S    


# generate G matrix
def makeG(f,d):
    G = np.array([[1         , d[1]      , d[3]      , d[1]*d[3]], 
                  [d[0]      , f[0]      , d[0]*d[3] , f[0]*d[3]], 
                  [d[2]      , d[1]*d[2] , f[1]      , f[1]*d[1]], 
                  [d[0]*d[2] , f[0]*d[2] , f[1]*d[0] , f[0]*f[1]]])
    return G

# generate F matrix
def makeF(omega):
    c = np.float(np.cos(omega))
    s = np.float(np.sin(omega))
    F = np.array([[  c*c,  c*s, -c*s, -s*s],
                  [ -c*s,  c*c,  s*s, -c*s],
                  [  c*s,  s*s,  c*c,  c*s],
                  [ -s*s,  c*s, -c*s,  c*c]])
    F = F.T
    return F

# estimate S from M, see 6a-6c of Shauns/Marks paper
def estimatedS(omegaEstimate, M):
    c = np.float(np.cos(omegaEstimate))
    s = np.float(np.sin(omegaEstimate))
    ES = np.array([c*c*M[0] + c*s*(M[2]-M[1]) - s*s*M[3],   
                   (M[1] + M[2])/2,                         
                   -s*s*M[0] + c*s*(M[2]-M[1]) + c*c*M[3]])
    return ES
  
  
def getBiomass(S,N,f,d,omega,A,p):
    noS = np.shape(S)[1]
    G = makeG(f,d)
    Ghat = np.diag((1,1,1,1))
    F = makeF(omega)
    M = G@F@S + N
    # originally, the omega is estimated using bickle/bates:
    # OE = bickleBates(M, omega)
    # this step is skipped because the estimation makes no difference
    # just a test where omega ist estimated perfectly
    
    OE = 0
    #OE = omega
    ES = estimatedS(OE,M)
    biomass = powerLaw(np.sum(abs(ES[1])**2)/noS,A,p)
    return biomass

# calculates mean, 2 sigma, and 3 sigma confidence interval for one ROI 
# takes the sigmas in natural values
def agb_error_stats(noS,S,Sbiomass,noCI,A_pl,p_pl,
                    NESZ,CI_sd,CI_amp_comp_corr,CI_arg_comp_corr,XT_sd,XT_amp_comp_corr,
                    XT_arg_comp_corr,PBE):
  
  N = makeN(NESZ,4,noS)
  
  be = np.zeros(noCI)
  
  for i in np.arange(noCI):
    # new policy: ignore omega 
    omega = 0.0
    #omega = np.random.rand(1)*2*np.pi-np.pi
    d = errorterm("xt",XT_sd ,XT_amp_comp_corr,XT_arg_comp_corr,1)
    f = 1 + errorterm("ci",CI_sd,CI_amp_comp_corr, CI_arg_comp_corr,1)
    biomass = getBiomass(S,N,f,d,omega, A_pl, p_pl)
    # the biomass error
    be[i] = np.mean(100*(biomass-Sbiomass)/Sbiomass)
  # for the mean and variance, the absolute is not necessary, 
  # for the CIs it simply means the percentage of being below +/-X % error
  mean_be = np.mean(be)
  variance = np.var(be)
  perc_9545 = np.percentile(np.abs(be),95.45)
  perc_9973 = np.percentile(np.abs(be),99.73)
  
  
  #### addition: get the probability that error < PBE:
  # set up kernel density estimation
  linspace1 = np.linspace(np.min(be), np.max(be), 10000)
  kde = stats.gaussian_kde(be)
  ys = kde.evaluate(linspace1)
  ys_cum = np.cumsum(ys)
  ys_cum_norm = ys_cum/ys_cum[-1]
  # get the probs of be < -20 and be > 20
  lowval = ys_cum_norm[(linspace1+PBE).argmin()]
  highval = ys_cum_norm[np.abs(linspace1-PBE).argmin()]
  prob = highval-lowval
  if prob > 1.0:
      prob=1.0
  
  return mean_be, perc_9545, perc_9973, variance, prob
  
  
def plot_ci_xt(sd,amp_comp_corr,arg_comp_corr,noS, fontsize, filename):
  
  checka = 1
  checkb = 0
  while checka != checkb:
  
    samples = errorterm("xt",sd ,amp_comp_corr,arg_comp_corr,noS)
    var1 = samples[1]
    var2 = samples[3]
    if np.round(20*np.log10(np.percentile(np.abs(var1),99.73)),2) == np.round(20*np.log10(np.percentile(np.abs(var2),99.73)),2):
      checkb = 1
  
  phasediff = np.angle(var1) - np.angle(var2)
  phasediff[phasediff>np.pi] = phasediff[phasediff>np.pi]-2*np.pi
  phasediff[phasediff<-np.pi] = phasediff[phasediff<-np.pi]+2*np.pi
  phase_corr1 = np.angle(np.corrcoef(var1,var2)) 
  
  # Plot
  lim = 4*sd
  fig1 = plt.figure(1, (5,5))
  ax11 = fig1.add_subplot(221)
  ax11.plot(np.real(var1),np.imag(var1),"x", markersize = 0.5)
  ax11.set_xlim(-lim,lim)
  ax11.set_ylim(-lim,lim)
  ax11.grid()
  ax11.set_xlabel("real", fontsize = fontsize+2)
  ax11.set_ylabel("imaginary", fontsize = fontsize+2)
  ax11.set_title(r"$\delta_1$", fontsize = fontsize+4)
  perc_9973 = np.percentile(np.abs(var1),99.73)
  ax11.plot(np.cos(np.linspace(0,360,1000)*np.pi/180)*perc_9973, np.sin(np.linspace(0,360,1000)*np.pi/180)*perc_9973, "g-", linewidth = 1.0, label = "|x|= " + np.str(np.round(perc_9973,5)) + " (" +np.str(np.round(20*np.log10(perc_9973),2)) + " dB)")
  plt.legend(fontsize = fontsize, loc = "lower right")
  
  ax11.tick_params(axis='both', which='major', labelsize=fontsize)
  
  ax12 = fig1.add_subplot(222)
  ax12.plot(np.real(var2), np.imag(var2),"x", markersize = 0.5)
  ax12.set_xlim(-lim,lim)
  ax12.set_ylim(-lim,lim)
  ax12.grid()
  ax12.set_xlabel("real", fontsize = fontsize+2)
  ax12.set_ylabel("imaginary", fontsize = fontsize+2)
  ax12.set_title(r"$\delta_3$", fontsize = fontsize+4)
  perc_9973 = np.percentile(np.abs(var2),99.73)
  ax12.plot(np.cos(np.linspace(0,360,1000)*np.pi/180)*perc_9973, np.sin(np.linspace(0,360,1000)*np.pi/180)*perc_9973, "g-", linewidth = 1.0, label = "|x|= " + np.str(np.round(perc_9973,5)) + " (" +np.str(np.round(20*np.log10(perc_9973),2)) + " dB)")
  plt.legend(fontsize = fontsize, loc = "lower right")
  
  ax12.tick_params(axis='both', which='major', labelsize=fontsize)
  
  #fig2 = plt.figure(2)
  ax13 = fig1.add_subplot(212)
  ax13.grid()
  ax13.hist(phasediff,50)
  ax13.set_xlabel("phase difference [rad]", fontsize = fontsize+2)
  ax13.set_ylabel("frequency", fontsize = fontsize+2)
  ax13.set_title("Histogram of phase differences", fontsize = fontsize+2)
  low,high = plt.ylim()
  ax13.plot([phase_corr1[0,1],phase_corr1[0,1]],[low,high],color = "red", linestyle = "dotted", linewidth = 2, label = "$\\angle$$\\rho$")
  #ax21.plot([np.median(phasediff),np.median(phasediff)],[low,high],color = "orange", linestyle = "dotted", linewidth = 2, label = "Median")
  plt.legend( fontsize = fontsize+2)
  
  ax13.tick_params(axis='both', which='major', labelsize=fontsize)
  
  plt.tight_layout()
  
  fig1.savefig(filename, dpi = 300)