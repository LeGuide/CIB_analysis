# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:32:42 2019

@author: Guido Riembauer
"""
import numpy as np

###############################################################################
# ADAPTED:
# New function that checks -kz0/+kz0 as well as the num
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
  
  # get the minimums and maximums of the modified kz stack, return as a 2D array
  min_array = np.nanmin(kz_stack, axis = 2)
  max_array = np.nanmax(kz_stack, axis = 2)
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
        # within the kz interval of the respective master-slaves combination
        #######################################################################
        elif master == -2:
            num_pixels_kz = []
            for kzIdx in np.arange(kz_stack.shape[2]):
                TEMP = kz_stack[:,:,kzIdx].reshape((Nr, Nc, 1))
                opt_str.master = kzIdx
                # check the number of pixels for which the kz0 criterion holds
                pixelcount = kz0_crit(kz_stack-TEMP, opt_str)
                num_pixels_kz.append(pixelcount)
                #print("For Master index {} of pol {} the number of pixels that satisfy the criterion is {}".format(kzIdx,pol_ind, pixelcount))
            # get the index of the max
            max_idx = num_pixels_kz.index(np.max(num_pixels_kz))
            print("Choosing index {} as optimal master SLC for pol {}".format(max_idx,pol_ind))
            # the rest is as in the normal approach where the master is defined  (see below)
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