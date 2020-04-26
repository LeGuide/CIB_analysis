# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:51:56 2019
This is a module that simply comprises the required algorithms from the MAAP.
They are redefined to fit the downloaded data format
@author: Guido Riembauer
"""

import numpy as np
from scipy.signal import medfilt2d
from numpy.matlib import repmat
import gdal
from gdalconst import GA_ReadOnly
from scipy.signal import convolve2d
from scipy.interpolate import interp1d, griddata, interpn
from scipy.stats import linregress
import pdb
import os
import rasterstats
import rasterio
import geopandas as gpd
import os

#%% slope calculation


def calcslope(pixel_spacing, z_flight, SLR_start, dem):
    
    dr = pixel_spacing
    H0 = z_flight
    R0 = SLR_start
    Nr,Na = dem.shape
    tmp_offnad = np.arccos((H0-dem)/repmat((R0+dr*np.arange(Nr)).reshape((Nr, 1)), 1, Na))
    # Initialise structures:
    angle1 = np.full((Nr,Na), np.NaN, dtype=dem.dtype)
    angle2 = np.full((Nr,Na), np.NaN, dtype=dem.dtype)

    # Computation of local slope:
    tmp_offnad = tmp_offnad.T
    dem = dem.T
    angle1[1:-1, :] = np.arctan2(dem[:, 1:-1] - dem[:, :-2], dr/np.sin(tmp_offnad[:, 1:-1]) + (dem[:, 1:-1] - dem[:, :-2])*np.tan(tmp_offnad[:, 1:-1])).T
    angle2[1:-1, :] = np.arctan2(dem[:, 2:] - dem[:, 1:-1], dr/np.sin(tmp_offnad[:, 1:-1]) + (dem[:, 2:] - dem[:, 1:-1])*np.tan(tmp_offnad[:, 1:-1])).T
    angle = (angle1 + angle2)/2
    tmp_offnad = tmp_offnad.T
    
    # Filter angle map:
    angle = medfilt2d(angle,kernel_size =5)

    return angle, tmp_offnad

#%% ground notching
    
###############################################################################
# adapted: new function that checks the -kz0/+kz0 criterion as well as the num
# of pixels for which the kz0 is within the boundaries
###############################################################################
def kz0_crit(kz_stack, opt_str):
  # Check if it is better to generate +kz0 or -kz0
  if np.nansum((np.nanmax(kz_stack, axis=2) >= opt_str.kz0) &\
               (np.nanmin(kz_stack, axis=2) <= opt_str.kz0)) < np.nansum(\
               (np.nanmax(kz_stack, axis=2) >= -opt_str.kz0) &\
               (np.nanmin(kz_stack, axis=2) <= -opt_str.kz0)):
      opt_str.kz0 = -opt_str.kz0
  test_kz0 = opt_str.kz0
  
  # get the minimums and maximums of the modified kz stack
  min_array = np.nanmin(kz_stack, axis = 2)
  max_array = np.nanmax(kz_stack, axis = 2)
  #pdb.set_trace()
  # check for how many pixels the kz0 is in between
  num_array = (min_array < test_kz0) & (test_kz0 < max_array)
  pixelcount = np.sum(num_array)
  return pixelcount
  
  
def GroundNotching(I, kz_stack, opt_str):
    """
    % It computes the "ground notched" slc.
    % 
    % INPUT
    %      I: [Nr x Nc x N] stack of calibrated, ground steered slc images.
    %      kz_stack: [Nr x Nc x N] array of phase-to-height conversion
    %                   factors (kz) OR [N x 1] vector of kz for constant
    %                   geometries. Needed if N > 2.
    %      opt_str.
    %                  kz0: desired phase-to-height
    %                  master: index of the master image
    %                  z_demod: (optional. Default: 0) vertical spectrum
    %                           demodulation height to perform interpolation
    %                  model_sign: (optional. Default: +1) model for the
    %                              interferometric phase: model_sign*kz*z
    %                  
    % OUTPUT
    %       GroundNotchedSLC: [Nr x Nc] ground notche slc
    %
    %    % DEPENDENCIES
    %             kzInterp
    """
    Nr, Nc, N = I.shape
    
    if N == 2:
        
        GroundNotchedSLC = I[:, :, 0] - I[:, :, 1]
        varargout = np.zeros((Nr, Nc), dtype=np.int8)
        
    else:
        
        if not(hasattr(opt_str, 'kz0')) or not(hasattr(opt_str, 'master')):
            print('Error in GroundNotching. Invalid number of arguments.\n')
            GroundNotchedSLC = []
            return
        # Demodulation elevation
        if not(hasattr(opt_str, 'z_demod')):
            opt_str.z_demod = np.pi/opt_str.kz0/2
        # Check if it is better to generate +kz0 or -kz0
        if np.nansum((np.nanmax(kz_stack, axis=2) >= opt_str.kz0) &\
                (np.nanmin(kz_stack, axis=2) <= opt_str.kz0)) < np.nansum(\
                (np.nanmax(kz_stack, axis=2) >= -opt_str.kz0) &\
                (np.nanmin(kz_stack, axis=2) <= -opt_str.kz0)):
            opt_str.kz0 = -opt_str.kz0
        
        Ikz0, varargout = kzInterp(I, kz_stack, opt_str)
        
        GroundNotchedSLC = I[:, :, opt_str.master] - Ikz0
        
    return GroundNotchedSLC, varargout

def kzInterp(I, kz_stack, opt_str):
    """It generates a synthetic slc by interpolating the stack of slc "I"
    % defined over the kz axis specified by kz_stack in correspondence of the
    % desired kz "kz0".
    % 
    % INPUT
    %      I: [Nr x Nc x N] stack of slc
    %      kz_stack: [Nr x Nc x N] stack of phase-to-height conversion factors
    %      opt_str.
    %              kz0: desired phase-to-height
    %              z_demod: (optional. Default: 0) vertical spectrum
    %                       demodulation height to perform interpolation
    %              model_sign: (optional. Default: +1) model for the
    %                          interferometric phase model_sign*kz*z
    % 
    % OUTPUT
    %       Ikz0: [Nr x Nc] synthetic slc
    %       varargout{1}: [Nr x Nc] logical mask true if the desired kz is out
    %                     of the available range"""
    if not(hasattr(opt_str, 'kz0')):
        Ikz0 = []
        print('Error in kzInterp. The desired kz must be specified.\n')
        return
    if not(hasattr(opt_str, 'z_demod')):
        z_demod = 0
    else:
        z_demod = opt_str.z_demod
        if not(hasattr(opt_str, 'model_sign')):
            model_sign = +1
        else:
            model_sign = opt_str.model_sign
    
    Nr, Nc, N = I.shape
    # Demodulation
    #pdb.set_trace()
    I = I*np.exp(-1j*model_sign*kz_stack*z_demod)
    
    # Linear interpolation
    pre_kz_ind = np.zeros((Nr, Nc), dtype=np.int8)
    post_kz_ind = np.zeros((Nr, Nc), dtype=np.int8)
    pre_kz_abs_diff = np.zeros((Nr, Nc))+np.inf
    post_kz_abs_diff = np.zeros((Nr, Nc))+np.inf
    
    for n in np.arange(N):
        curr_kz_diff = kz_stack[:, :, n] - opt_str.kz0
        curr_kz_abs_diff = np.abs(curr_kz_diff)
        
        pre_kz_mask = curr_kz_diff < 0
        post_kz_mask = np.logical_not(pre_kz_mask)
        
        # To Be Replaced
        pre_tbr = (np.abs(curr_kz_diff) < pre_kz_abs_diff) & pre_kz_mask
        post_tbr = (np.abs(curr_kz_diff) < post_kz_abs_diff) & post_kz_mask
        
        pre_kz_ind[pre_tbr] = n
        post_kz_ind[post_tbr] = n
        
        pre_kz_abs_diff[pre_tbr] = curr_kz_abs_diff[pre_tbr]
        post_kz_abs_diff[post_tbr] = curr_kz_abs_diff[post_tbr]
        
    # Desired kz_stack out of range (To Be Extrapolated)
    pre_tbe = np.isinf(pre_kz_abs_diff)
    post_tbe = np.isinf(post_kz_abs_diff)
    
    pre_kz_ind[pre_tbe] = 0
    post_kz_ind[post_tbe] = N-1
    
    [C, R] = np.meshgrid(np.arange(Nc), np.arange(Nr))
    
    kz_pre = kz_stack[R, C, pre_kz_ind]
    kz_post = kz_stack[R, C, post_kz_ind]
    frac_part = (opt_str.kz0 - kz_pre)/(kz_post - kz_pre)
    
    Ikz0 = (1 - frac_part)*I[R, C, pre_kz_ind] + frac_part*I[R, C, post_kz_ind]
    
    Ikz0[pre_tbe | post_tbe] = np.spacing(1)
    
    # Modulation
    Ikz0 = Ikz0*np.exp(1j*model_sign*opt_str.kz0*z_demod)
    
    varargout = pre_tbe | post_tbe
    
    return Ikz0, varargout
def ground_notching(I, kz_stack, master, z_emph):
    # Ground notching
    GroundNotchedSLC = list()
        
    # Desired elevation for the peak of the ground notching processing
    class opt_str:
        pass
    opt_str.z_demod = z_emph/2 # [m]
    opt_str.kz0 = np.pi/opt_str.z_demod/2
    # Polarimetric channel
    for pol_ind in np.arange(len(I)):
        Nr, Nc, N = I[0].shape
        notch_final = np.zeros((Nr, Nc))
        mask_final = np.zeros((Nr, Nc))
        if N == 2:
            opt_str.master = master
            notch_final, mask_final = GroundNotching(I[pol_ind], kz_stack, opt_str)
            GroundNotchedSLC.append(notch_final)
        elif master == -1: # multi-master
            for kzIdx in np.arange(kz_stack.shape[2]):
                TEMP = kz_stack[:,:,kzIdx].reshape((Nr, Nc, 1))
                opt_str.master = kzIdx
                notch_temp, notch_mask = GroundNotching(I[pol_ind], kz_stack-TEMP, opt_str)
                notch_temp[notch_mask == 1] = 0
                #pdb.set_trace()
                notch_final = notch_final + np.abs(notch_temp)**2
                mask_final = mask_final + 1 - notch_mask
            mask_final[mask_final == 0] = 1
            GroundNotchedSLC.append(np.sqrt(notch_final / mask_final))
        #######################################################################
        # ADAPTED:
        # choose optimal master from a stack. This is based on whether the kz0 is
        # within the kz limits of the respective master-slaves combination
        #######################################################################
        elif master == -2:
            #pdb.set_trace()
            num_pixels_kz = []
            for kzIdx in np.arange(kz_stack.shape[2]):
                TEMP = kz_stack[:,:,kzIdx].reshape((Nr, Nc, 1))
                opt_str.master = kzIdx
                # check the number of pixels for which the kz0 criterion holds
                pixelcount = kz0_crit(kz_stack-TEMP, opt_str)
                num_pixels_kz.append(pixelcount)
                print("For Master index {} of pol {} the number of pixels that satisfy the criterion is {}".format(kzIdx,pol_ind, pixelcount))
            # get the index of the max
            max_idx = num_pixels_kz.index(np.max(num_pixels_kz))
            print("Choosing index {} as optimal master SLC for pol {}".format(max_idx,pol_ind))
            # the rest is as in the normal approach where the master is indicated (see below)
            opt_str.master = max_idx
            TEMP = kz_stack[:,:,opt_str.master].reshape((Nr, Nc, 1))
            notch_final, mask_final = GroundNotching(I[pol_ind], kz_stack-TEMP, opt_str)
            notch_final[mask_final == 1] = 0
            GroundNotchedSLC.append(notch_final)
        else:
            opt_str.master = master
            TEMP = kz_stack[:,:,opt_str.master].reshape((Nr, Nc, 1))
            notch_final, mask_final = GroundNotching(I[pol_ind], kz_stack-TEMP, opt_str)
            notch_final[mask_final == 1] = 0
            GroundNotchedSLC.append(notch_final)
    return GroundNotchedSLC

#%% geocoding
    
def GrdToSlrProj(grd_image, Azimuth, Range, Nr, Na):
    #pdb.set_trace()
    'Projection of an image from Slant Range geometry to Ground Projected geometry'
    # Mask of the data inside the GRD projected image:
    mask = 1-np.isnan(grd_image)
    # Create an empty image of NaN:
    slrFile = np.full((Nr, Na), np.NaN, dtype=grd_image.dtype)
    # Project the image in the slant range geometry:
    Range[Range>=Nr] = np.nan
    Azimuth[Azimuth>=Na] = np.nan
    # fix cast
    slrFile[Range[mask==1].astype('uint16'), Azimuth[mask==1].astype('uint16')] = grd_image[mask==1]
    mask = 1-np.isnan(slrFile)
    xx, yy = np.meshgrid(np.arange(slrFile.shape[1]), np.arange(slrFile.shape[0]))
    f = griddata(np.array([xx[mask==1], yy[mask==1]]).T, slrFile[mask==1], np.array([xx.flatten(), yy.flatten()]).T)
    slrFile = f.reshape(yy.shape)
    #slrFile = medfilt2d(slrFile,kernel_size =11)
    ####################################################
    # ADAPTION
    # reduce size of filter kernel to speed up process
    #slrFile = medfilt2d(slrFile,kernel_size =3)
    
    return slrFile



def SlrToGrdProj(slr_image, Azimuth, Range):
    'Projection of an image from Slant Range geometry to Ground Projected geometry'
    # Project the image in the ground projected geometry:
    Nr, Na = slr_image.shape
    f = interpn((np.arange(Nr),np.arange(Na)), slr_image, np.array([Range.flatten(), Azimuth.flatten()]).T, bounds_error=0)
    grd_image = f.reshape(Range.shape)
    
    return grd_image
    
def save_tiff(image_matrix,ref_geotransform_file,outputfile):
    if isinstance(image_matrix,list) : #if multiband list image
        nbband=len(image_matrix)
        rasterxsize=image_matrix[0].shape[1]   
        rasterysize=image_matrix[0].shape[0]
    elif isinstance(image_matrix,np.ndarray): #if one band image
        nbband=1
        rasterxsize=image_matrix.shape[1]
        
        rasterysize=image_matrix.shape[0]
        
    Range_driver = gdal.Open(ref_geotransform_file, GA_ReadOnly)   
    outdriver = gdal.GetDriverByName('GTiff')
    grd_image_driver = outdriver.Create(outputfile, rasterxsize,rasterysize,nbband, gdal.GDT_Float64)
    grd_image_driver.SetGeoTransform(Range_driver.GetGeoTransform())
    grd_image_driver.SetProjection(Range_driver.GetProjection())
    
    if nbband > 1 :
        for band in range(nbband):
        
            # Save the corresponding band of the image in the ground projected geometry:
            grd_image_driver.GetRasterBand(band+1).WriteArray(image_matrix[band])
    else : 
        grd_image_driver.GetRasterBand(1).WriteArray(image_matrix)
    
    grd_image_driver = None

#%% sigma0
    
def compute_sigma0(input_matrix, theta, nlook, campaign, surface_resol) :
    # Computation of Sigma0 (natural) as given in the corresponding campaign reports:
    sigma0 = convolve2d(np.absolute(input_matrix)**2*np.tan(theta),np.ones((nlook, nlook))/nlook/nlook,mode='same')
    # Additional calibration necessary for tropisar, biosar3 and afrisar_onera:
    if campaign in ['tropisar', 'biosar3', 'afrisar_onera']:
        sigma0 /= float(surface_resol)
    sigma0[sigma0 <= 0] = np.NaN
    return sigma0
  
def compute_sigma0_stack(input_stack, theta, nlook, campaign, surface_resol) :
    # This function is an adaption of the above one as it takes one complete ground cancelled four polarization stack
    # and computes the sigma0 of HV and VH as the combination of both, so the output stack has only 3 pols
    # Computation of Sigma0 (natural) as given in the corresponding campaign reports
    sigma0_list = []
    for idx, image in enumerate(input_stack):
      if idx == 1:
        sigma0 = convolve2d(((np.absolute(image)+np.absolute(input_stack[idx+1]))/2)**2*np.sin(theta),np.ones((nlook, nlook))/nlook/nlook,mode='same')
      if idx == 2:
        continue
      else:
        sigma0 = convolve2d(np.absolute(image)**2*np.sin(theta),np.ones((nlook, nlook))/nlook/nlook,mode='same')
      # Additional calibration necessary for tropisar, biosar3 and afrisar_onera:
      if campaign in ['tropisar', 'biosar3', 'afrisar_onera']:
        sigma0 /= float(surface_resol)
      sigma0[sigma0 <= 0] = np.NaN
      sigma0_list.append(sigma0)
    return sigma0_list
  
### estimate complex covariance
    
def comp_cov(input_matrix1, input_matrix2,theta, campaign, surface_resol) :
    comp_cov = input_matrix1 * np.conjugate(input_matrix2) * np.tan(theta)
    if campaign in ['tropisar', 'biosar3', 'afrisar_onera']:
      comp_cov /= float(surface_resol)
    return comp_cov
  
#%%

def get_stats(input_raster,input_path):
  pass

def fit_pl(input_raster,input_path):
    # takes an in put raster and a path to shapefiles as inputs and returns 
    # power law parameters a and b as in the Schlund paper. It is expected
    # that the shapefiles have a column named "agb_loc" where the reference
    # agb value is stored
    #os.environ["PROJ_LIB"] = "C:/Anaconda3/envs/test36/Library/share"
    # get the crs of the raster 
    raster_crs = rasterio.open(input_raster).crs.data
    
    refagb = []
    backscatter = []
    
    # Loop through the folder with the shapes
    for item in os.listdir(input_path):
      if item.endswith('.shp'):
        filepath = input_path+item
        
        # reproject to raster's crs
        shapefile = gpd.read_file(filepath)
        shapefile = shapefile.to_crs(raster_crs)
        
        # Get the entry from the column agb_loc
        refagb.append(shapefile.agb_loc[0])
        
        # use rasterstats to extract the backscatter intensity
        stats = rasterstats.zonal_stats(shapefile,input_raster, nodata = -999)
        mean_bs = stats[0]['mean']
        backscatter.append(mean_bs)
    
    # remove entries in both lists where elements are missing in the other list
    backscatter_arr = []
    refagb_arr = []
    for idx,elem in enumerate(backscatter):
      if elem is not None:
        if elem != 0.0:
          if refagb[idx] != 0.0:
            backscatter_arr.append(elem)
            refagb_arr.append(refagb[idx])
    refagb = refagb_arr
    backscatter = backscatter_arr
    backscatter_log = 10*np.log10(backscatter)

    ###
    ###############################################################################
    # fit the power law parameters a and b to the data
    ###############################################################################

    # generate observations vector and design matrix
    y = np.transpose(np.matrix(backscatter_log))
    A = np.transpose(np.matrix((np.ones(len(backscatter_log)),np.log10(refagb))))
    
    slope,intercept,rvalue,pvalue,stderr = linregress(np.log10(refagb),backscatter_log)
    
    listlinregress = [slope,intercept,rvalue,pvalue,stderr,rvalue**2]
    # Estimate a, b 
    #A_trans = np.asarray(np.transpose(A))
    
    # least squares approach, the system is in ax+b now because refagb is in log space too
    xhat = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@y
    
    b = np.double(xhat[0])
    a = np.double(xhat[1])
    
    return a,b, refagb, backscatter_log,listlinregress
  
  
    