# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:22:10 2019
This script loads all necessary data for ground-cancellation and performs the
ground cancellation based on a chosen emphasized height above the ground.
The master image for ground cancellation is chosen by the user or automatically
selected by the algorithm. 
@author: Guido Riembauer
"""
import numpy as np
import gdal
from maap_processing import calcslope
from maap_processing import GrdToSlrProj
from maap_processing import SlrToGrdProj
from maap_processing import ground_notching
from maap_processing import compute_sigma0
#from maap_processing import compute_sigma0_stack
from maap_processing import comp_cov
from maap_processing import save_tiff
from maap_processing import fit_pl
import matplotlib.pyplot as plt
#import pdb

###############################################################################
# Define Input:
# SLCs; kzs, DTM, inc, az, rg
# important: use ALL four polarisations, HV and VH will be merged later
###############################################################################
folder = "D:/6Mh_Polimi_Aresys/Lope/afrisar_dlr/"
#folder = "D:/6Mh_Polimi_Aresys/Paracou/tropisar/"
resultfolder = "D:/6Mh_Polimi_Aresys/code/results/"
#####
# indicate prefix in the filenames
prefix = "afrisar_dlr"
#prefix = "tropisar_"
#####
# indicate the master image
master = prefix + "T2-0"
#master = prefix + "C402"
#master = prefix + "CH2-0"
#master = prefix + "CH4-0"
#master = prefix + "CH1-0"
#####
# indicate the images in the stack to be used
slcs = [master, prefix+"T2-1", prefix+"T2-2", prefix+"T2-3", prefix+"T2-4", prefix+"T2-5"]
#slcs = [master, prefix+"CH1-1", prefix+"CH1-2"]
#slcs = [master, prefix+"CH4-1", prefix+"CH4-2"]
#slcs = [master, prefix+"C403", prefix+"C404", prefix+"C405",prefix+"C406",prefix+"C407"]
#slcs = [master, prefix+"CH2-1", prefix+"CH2-2"]
#slcs = [master, prefix + "CH4-0", prefix+"CH4-1", prefix+"CH4-2"]
#####
# indicate the DEM suffix
dtm = master+"_dem_dem.tiff"
#####
# this is where the reference ROI shapefiles are stored
#ROI_path = "C:/3_BIOMASS/L2/Data/Insitu/agb_attached_labriere/Mabounie/1ha/"
ROI_path = "C:/3_BIOMASS/L2/Data/Insitu/agb_attached_labriere/Lope/1ha/"
#ROI_path = "C:/3_BIOMASS/L2/Data/Insitu/agb_attached_labriere/Paracou/6.25ha/"
#####
# indicate the height above ground to be emphasized
notch_emph_height = 50
# number of looks to get sigma0
nlook = 9

# indicate whether the process should also be done for non-notched data
reference_fit = False

# indicate whether the master has a kz (hence is actually not the naster the kzs have been calculated from):
master_kz = False

# indicate whether a DEM is required (only if there is no inc file):
dem_required = True

###########################################
# these are taken from the campaign xml file
if master == prefix + "CH2-0":
  pixel_spacing = 1.1988876
  surface_resol = np.nan
  z_flight = 6384.60
  heading = 230
  SLR_start = 6579.4951
  
elif master == prefix + "CH1-0":
  pixel_spacing = 1.1988876
  surface_resol = np.nan
  z_flight = 6384.44
  heading = 124
  SLR_start = 6579.4951

elif master == prefix + "CH4-0":
  pixel_spacing = 1.1988876
  surface_resol = np.nan
  z_flight = 6383.12
  heading = 320
  SLR_start = 6579.4951
  
elif master == prefix + "T2-0":
  pixel_spacing = 1.1988876
  surface_resol = np.nan
  z_flight = 6383.36
  heading = 320
  SLR_start = 6536.3352

elif master == prefix + "CMA2-3":
  pixel_spacing = 1.1988876
  surface_resol = np.nan
  z_flight = 6152.36
  heading = 0.0
  SLR_start = 6536.3352

elif master == prefix + "C402":
  pixel_spacing = 1.0
  surface_resol = 2.18525
  z_flight = 3904.8
  heading = 8.1
  SLR_start = 4255.0
  z_terrain = -110.1
#%%############################################################################
# get all required paths 
###############################################################################
I_hh = []
I_hv = []
I_vh = []
I_vv = []
kz = []
for idx,slc in enumerate(slcs):
  # only if the master itself has a kz too, otherwise take only the others
  if master_kz:
    file_kz = gdal.Open(folder+slc+"_kz.tiff")
    data_kz = file_kz.ReadAsArray()
    kz.append(data_kz)
    file_kz = None
    print("reading {} done".format(folder+slc+"_kz"))
  else:
    if idx>0:
      file_kz = gdal.Open(folder+slc+"_kz.tiff")
      data_kz = file_kz.ReadAsArray()
      kz.append(data_kz)
      file_kz = None
      print("reading {} done".format(folder+slc+"_kz"))
    
  for pidx,pol in enumerate(["_SLC_HH.tiff","_SLC_HV.tiff","_SLC_VH.tiff","_SLC_VV.tiff"]):
    file = gdal.Open(folder+slc+pol)
    data = file.ReadAsArray()
    if pidx == 0:
      I_hh.append(data)
    elif pidx == 1:
      I_hv.append(data)
    elif pidx == 2:
      I_vh.append(data)
    elif pidx == 3:
      I_vv.append(data)
    file = None
    print("reading {} done".format(folder+slc+pol))

I_hh = np.asarray(I_hh)
I_hv = np.asarray(I_hv)
I_vh = np.asarray(I_vh)
I_vv = np.asarray(I_vv)

I_hh = I_hh.transpose(1,2,0)
I_hv = I_hv.transpose(1,2,0)
I_vh = I_vh.transpose(1,2,0)
I_vv = I_vv.transpose(1,2,0)

kz = np.asarray(kz)
kz = kz.transpose(1,2,0)
# add another kz = 0 array for the master
if master_kz == False:
  zeros = np.zeros(np.shape(kz[:,:,0]))
  kz = np.dstack((zeros,kz))


# read further ancillary data
  
if dem_required:
  dem_file = gdal.Open(folder+dtm)
  dem_data = dem_file.ReadAsArray()
  dem_file = None
  print("reading {} done".format(folder+dtm))

if dem_required == False:
  inc_file = gdal.Open(folder+master+"_inc.tiff")
  inc_data = inc_file.ReadAsArray()
  inc_file = None
  print("reading {} done".format(folder+master+"_inc.tiff"))

az_file = gdal.Open(folder+master+"_az.tiff")
az_data = az_file.ReadAsArray()
az_data = az_data.astype("float")
az_file = None
print("reading {} done".format(folder+master+"_az.tiff"))

rg_file = gdal.Open(folder+master+"_rg.tiff")
rg_data = rg_file.ReadAsArray()
rg_data = rg_data.astype("float")
rg_file = None
print("reading {} done".format(folder+master+"_rg.tiff"))

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Reading data done")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#%%############################################################################
# calculate theta
###############################################################################
if prefix == "tropisar_":
  print("calculating theta...")
  # project dem to slant range to perform slope/theta calculation
  #dem_slr = GrdToSlrProj(dem_data, az_data, rg_data, np.shape(I_hh)[1], np.shape(I_hh)[0])
  
  # if the files are too large, taking only every other value does the trick
  dem_slr = GrdToSlrProj(dem_data[::3,::3], az_data[::3,::3], rg_data[::3,::3], np.shape(I_hh)[1], np.shape(I_hh)[0])
  # that way, the dem_slr axes are transposed, but later in the sigma0 step it works out
  angle, tmp_offnad = calcslope(pixel_spacing, z_flight, SLR_start, dem_slr)
  theta = tmp_offnad-angle
  # Transpose it to match the slcs and kzs
  theta = theta.T
else:
  #############
  # or: simply take the input file
  #############
  theta = inc_data

#%%############################################################################
# perform ground notching
###############################################################################
print("ground cancellation...")
I_complete = np.asarray([I_hh,I_hv,I_vh,I_vv])

I_gc = ground_notching(I_complete, kz, -2, notch_emph_height)

#%%############################################################################
# get sigma0 and the hh-vv complex covariance
###############################################################################
print("Covariance matrix terms estimation...")

#sigma0_gc = compute_sigma0_stack(I_gc, theta, nlook, prefix[:-1], surface_resol)
#sigma0_example = compute_sigma0_stack(I_complete[:,:,:,0], theta, nlook, prefix[:-1], surface_resol)


sigma0_gc = [compute_sigma0(pol,theta,nlook,prefix[:-1],surface_resol) for pol in I_gc]

# get hh vv covariance
hh_vv_comp_cov = comp_cov(I_gc[0],I_gc[3], theta, prefix[-1], surface_resol)
# split into real and imag parts
hh_vv_comp_cov_real = np.real(hh_vv_comp_cov)
hh_vv_comp_cov_imag = np.imag(hh_vv_comp_cov)

if reference_fit:
  sigma0_example = [compute_sigma0(pol,theta,nlook,prefix[:-1],surface_resol) for pol in I_complete[:,:,:,0]]

# remove the intermediate products
#I_gc = None
#I_complete = None
#I_nomaster = None
#I_hh = None
#I_hv = None
#I_vh = None
#I_vv = None
#kz = None
#hh_vv_comp_cov = None

#%%############################################################################
# geocode 
###############################################################################
# also here, rg and az have to be swapped
print("Geocoding...")
sigma0_gc_gr = [SlrToGrdProj(pol, rg_data, az_data) for pol in sigma0_gc]
if reference_fit:
  sigma0_example_gr = [SlrToGrdProj(pol, rg_data, az_data) for pol in sigma0_example]

hh_vv_comp_cov_real_gr = SlrToGrdProj(hh_vv_comp_cov_real, rg_data, az_data)
hh_vv_comp_cov_imag_gr = SlrToGrdProj(hh_vv_comp_cov_imag, rg_data, az_data)

### Temp: taking HV*2
sigma0_gc_gr[1] = sigma0_gc_gr[1]*2
sigma0_gc_gr[2] = sigma0_gc_gr[2]*2
if reference_fit:
  sigma0_example_gr[1] = sigma0_example_gr[1]*2
  sigma0_example_gr[2] = sigma0_example_gr[2]*2

#%%############################################################################
# save tiffs
###############################################################################

print("Saving tiffs...")
pollist = ["_HH_","_HV_","_VH_","_VV_"]

for idx,pol in enumerate(sigma0_gc_gr):
    save_tiff(pol, folder+master+"_az.tiff", resultfolder+master+ pollist[idx]+"gc.tiff")

save_tiff(hh_vv_comp_cov_real_gr, folder+master+"_az.tiff", resultfolder+master+ "_gc_HH_VV_cov_real.tiff")
save_tiff(hh_vv_comp_cov_imag_gr, folder+master+"_az.tiff", resultfolder+master+ "_gc_HH_VV_cov_imag.tiff")

if reference_fit:
  for idx,pol in enumerate(sigma0_example_gr):
    save_tiff(pol, folder+master+"_az.tiff", resultfolder+master+ pollist[idx]+"non_notched_example.tiff")

#%%########################################################################### #
# fit the notched and non-notched backscatter to the ref data
###############################################################################
print("Fitting power law to data...")

input_notched = resultfolder+master+ pollist[1]+"gc.tiff"
#input_notched = "C:/3_BIOMASS/L2/02_Software/Aresys_local testing/outdata/Lope_s0hv_pl_n10est1.tif"
#input_notched = "C:/3_BIOMASS/L2/02_Software/Aresys_local testing/outdata/Paracou_s0hv_4326pl_n10fix.tif"
#input_notched = "C:/3_BIOMASS/L2/02_Software/Aresys_local testing/outdata/Mabounie_s0hv_pl_n10est1.tif"
#input_notched = "D:/6Mh_Polimi_Aresys/code/results/tropisar_C402_VH_gc.tiff"
#input_notched = "D:/6Mh_Polimi_Aresys/code/results/afrisar_dlr_CMA2-3_HV_gc.tiff"

a_gc, b_gc, refagb_gc, backscatter_gc,listlinregress = fit_pl(input_notched, ROI_path)
if reference_fit:
  input_normal = resultfolder+master+ pollist[2]+"non_notched_example.tiff"
  a_norm, b_norm, refagb_norm, backscatter_norm = fit_pl(input_normal, ROI_path)

#%%############################################################################
# Plot the power law of notched vs. non notched backscatter
###############################################################################

fig1 = plt.figure(figsize = (8,6))
ax1 = fig1.add_subplot(111)
ax1.plot(refagb_gc, backscatter_gc, "x", color = "blue", label = "ground cancelled backscatter")
arange = np.arange(0,600)
range_bs_gc = a_gc*np.log10(arange)+b_gc
ax1.plot(arange, range_bs_gc, "-", color = "blue", label = " PL (ground cancelled)\na = {}\nb={}".format(np.round(a_gc,2),np.round(b_gc,2)))

if reference_fit:
  ax1.plot(refagb_norm, backscatter_norm, "x", color = "red", label = "normal backscatter")
  range_bs_norm = a_norm*np.log10(arange)+b_norm
  ax1.plot(arange, range_bs_norm, "-", color = "red", label = " PL (normal)\na = {}\nb={}".format(np.round(a_norm,2),np.round(b_norm,2)))

plt.legend()
ax1.set_xlim((0,600))
ax1.set_ylim((-25,-3))
ax1.grid()



