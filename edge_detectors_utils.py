#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_detectors_utils.py

"""

import numpy as np
from scipy.signal import correlate

def GaussianKernel1D(sigma):
    '''Returns 1D Gaussian kernel
    
    sigma: standard deviation of normal distribution
    '''
    
    kernel_size = 6 * sigma + 1;    
    kn = int((kernel_size - 1) / 2);
    X = np.arange(-kn, kn + 1, 1);
    
    kernel = np.exp(-(np.power(X, 2)) / (2 * sigma ** 2));
    kernel = kernel / kernel.sum();
    kernel = kernel.reshape(len(kernel), 1);
    
    return kernel;

def Denoising(img_in, sigma):
    '''Returns an image filtered with a Gaussian kernel
    
    sigma: standard deviation of normal distribution
    '''
    
    kernel = GaussianKernel1D(sigma);
    kernel_x = kernel;
    kernel_y = kernel.T;   
    
    img_out =  correlate(img_in, kernel_x, mode = 'same', method = 'auto');
    img_out =  correlate(img_out, kernel_y, mode = 'same', method = 'auto');

    # kernel = np.array([[2, 4, 5, 4, 2],
    #                     [4, 9, 12, 9, 4],
    #                     [5, 12, 15, 12, 5],
    #                     [4, 9, 12, 9, 4],
    #                     [2, 4, 5, 4, 2]]);
    # kernel = kernel / kernel.sum();
    # img_out =  correlate(img_in, kernel, mode = 'same', method = 'auto');
    
    return img_out;

def CalculateSobelResponse(img_in):
    '''Returns a tupil (magnitude, phase) sobel response of an image
    
    sigma: standard deviation of normal distribution
    '''    

    kernel_x1 = np.array([[1], [2], [1]]);
    kernel_x2 = np.array([[1, 0, -1]]);
    kernel_y1 = kernel_x1.T;
    kernel_y2 = -kernel_x2.T;

    gx =  correlate(img_in, kernel_x1, mode = 'same', method = 'auto');
    gx =  correlate(gx, kernel_x2, mode = 'same', method = 'auto');
    gy =  correlate(img_in, kernel_y1, mode = 'same', method = 'auto');
    gy =  correlate(gy, kernel_y2, mode = 'same', method = 'auto');
       
    magn = (gx ** 2 + gy ** 2) ** 0.5;
    phase = np.arctan2(gy, gx);
    
    return magn, phase;

def CalculateLaplacianResponse(img_in):
    '''Returns laplacian response of an image
    
    sigma: standard deviation of normal distribution
    '''     

    # Laplacian kernel with equally strong response for horizontal, vertical and diagonal edges:   
    kernel_laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]);
    img_out = correlate(img_in, kernel_laplacian, mode = 'same', method = 'auto');
    
    return img_out;

def CalculateDirEdgeResponse(img_in, orientation):
    '''Returns directional edge response of an image
    
    orientation: edge orientation
    '''      
    
    if orientation == 'hor': # horizontal edges
        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]);
    elif orientation == 'ver': # vertical edges
        kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]);
    elif orientation == '45deg': # + (plus) 45 degrees
        kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]);        
    elif orientation == 'm45deg': # - (minus) 45 degrees
        kernel = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]);        
  
    img_out = (correlate(img_in, kernel, mode = 'same', method = 'auto'));
    
    return img_out;        
        
def SobelEdgeDetector(img_in, kernel_sigma, threshold, noise_sigma):
    '''Returns a binary masks with horizontal and vertical edges
    
    img_in: original gray scale image
    kernel_sigma: Gaussian kernel standard deviation for denoising
    threshold: threshold for edge detection [0, 1]
    noise_sigma: noise standard deviation estimated from a flat image region
    '''      
    
    img_original = img_in;
    img_in = img_in / img_in.max(); # scaling 0 to 1
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    
    mask = np.zeros((nr, nc));
    img_in_denoised = Denoising(img_in, kernel_sigma);
    magn, _ = CalculateSobelResponse(img_in_denoised);
    temp = magn[10:-10, 10:-10];
    threshold = threshold * temp.max();

    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb_original = img_original[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_original = nb_original.reshape(3, 3);
            nb_original_sigma = np.std(nb_original);
            
            if magn[ii, jj] > threshold:
                mask[ii, jj] = 1;

            # noise supression
            if nb_original_sigma < noise_sigma:
                mask[ii, jj] = 0; 
                
    return mask;

def LaplacianEdgeDetector(img_in, kernel_sigma, threshold, noise_sigma):
    '''Returns a binary masks with horizontal, vertical and diagonal edges
    
    img_in: original gray scale image
    kernel_sigma: Gaussian kernel standard deviation for denoising
    threshold: threshold for edge detection [0, 1]
    noise_sigma: noise standard deviation estimated from a flat image region
    ''' 
    
    img_original = img_in;
    img_in = img_in / img_in.max(); # scaling 0 to 1
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    
    mask = np.zeros((nr, nc));
    img_in_denoised = Denoising(img_in, kernel_sigma);
    magn = CalculateLaplacianResponse(img_in_denoised);
    temp = magn[10:-10, 10:-10];
    threshold = threshold * temp.max();

    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb_original = img_original[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_original = nb_original.reshape(3, 3);
            nb_original_sigma = np.std(nb_original);
            nb_magn = magn[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_magn = nb_magn.reshape(3, 3);
            
            if magn[ii, jj] > threshold and (nb_magn <= 0).any():
                mask[ii, jj] = 1;
    
            # noise supression
            if nb_original_sigma < noise_sigma:
                mask[ii, jj] = 0; 
                
    return mask;

def SingleLineEdgeDetector(img_in, kernel_sigma, threshold, noise_sigma, orientation):
    '''Returns a binary masks with horizontal, vertical and diagonal edges
    
    img_in: original gray scale image
    kernel_sigma: Gaussian kernel standard deviation for denoising
    threshold: threshold for edge detection [0, 1]
    noise_sigma: noise standard deviation estimated from a flat image region
    orientation = edge orientation: 'hor' (horizontal), 'ver' (vertical), '45deg' (45 degrees), and 'm45deg' (- 45 degrees)
    ''' 
        
    img_original = img_in;
    img_in = img_in / img_in.max(); # scaling 0 to 1
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
    
    mask = np.zeros((nr, nc));
    img_in_denoised = Denoising(img_in, kernel_sigma);
    magn = CalculateDirEdgeResponse(img_in_denoised, orientation);
    temp = magn[10:-10, 10:-10];
    threshold = threshold * temp.max();

    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb_original = img_original[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_original = nb_original.reshape(3, 3);
            nb_original_sigma = np.std(nb_original);
            nb_magn = magn[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_magn = nb_magn.reshape(3, 3);
            
            if magn[ii, jj] > threshold and (nb_magn <= 0).any():
                mask[ii, jj] = 1;

            # noise supression
            if nb_original_sigma < noise_sigma:
                mask[ii, jj] = 0; 
                    
    return mask;

def CannyEdgeDetector(img_in, kernel_sigma, threshold, noise_sigma):
    '''Returns a binary masks with horizontal, vertical and diagonal edges
    
    img_in: original gray scale image
    kernel_sigma: Gaussian kernel standard deviation for denoising
    threshold: threshold for edge detection [0, 1]
    noise_sigma: noise standard deviation estimated from a flat image region
    orientation = edge orientation: 'hor' (horizontal), 'ver' (vertical), '45deg' (45 degrees), and 'm45deg' (- 45 degrees)
    ''' 
        
    # quantization directions (4 directions):
    # horizontal, vertical, plus 45 degrees, and minus 45 degrees

    d_array = np.array([[-22.5, 22.5, -157.5, 157.5],
                        [-112.5, -67.5, 67.5, 112.5],
                        [-157.5, -112.5, 22.5, 67.5],
                        [-67.5, -22.5, 112.5, 157.5]]);
            
    d_bool_array = np.zeros((4, 3, 3), dtype = np.bool);
    
    # horizontal direction
    d_bool_array[0] = np.array([[0, 0, 0],
                                [1, 0, 1],
                                [0, 0, 0]]);
    # vertical direction
    d_bool_array[1] = np.array([[0, 1, 0],
                                [0, 0, 0],
                                [0, 1, 0]]);   
    # plus 45 degrees direction
    d_bool_array[2] = np.array([[0, 0, 1],
                                [0, 0, 0],
                                [1, 0, 0]]);
    # minus 45 degrees direction
    d_bool_array[3] = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]);
    
    img_original = img_in;
    img_in = img_in / img_in.max(); # scaling 0 to 1
    
    nr, nc = img_in.shape;
    nr = int(nr);
    nc = int(nc);
        
    img_in_denoised = Denoising(img_in, kernel_sigma);
    magn, phase = CalculateSobelResponse(img_in_denoised);
    phase = 180 / np.pi * phase;
    phase_d = np.zeros((nr, nc), dtype = np.int8); # quantized phase
    
    # gradient vector phase quantization:
    # 0 = horizontal direction
    # 1 = vertical direction
    # 2 = plus 45 degrees direction
    # 3 = minus 45 degrees direction
    for ii in range(4):
        phase_mask = (((phase >= d_array[ii][0]) & (phase < d_array[ii][1])) \
            | ((phase >= d_array[ii][2]) & (phase < d_array[ii][3])));
        phase_d[phase_mask] = ii;

    mask = np.zeros((nr, nc));
    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb_magn = magn[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_magn = nb_magn.reshape(3, 3);
          
            # non-maximum suppresion
            temp_mask = d_bool_array[phase_d[ii, jj]];
            temp_val = nb_magn[temp_mask];     
            
            if (magn[ii, jj] < temp_val).any():
                mask[ii, jj] = 0;
            else:
                mask[ii, jj] = 1;
    
    magn = mask * magn;
    Tl = threshold[0] * magn.max();
    Th = threshold[1] * magn.max();            
    mask = np.zeros((nr, nc));
    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb_original = img_original[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_original = nb_original.reshape(3, 3);
            nb_original_sigma = np.std(nb_original);            
            nb_magn = magn[ii - 1:ii + 2, jj - 1:jj + 2];
            nb_magn = nb_magn.reshape(3, 3);
                          
            # double thresholding
            if magn[ii, jj] < Tl:
                mask[ii, jj] = 0; 
            elif magn[ii, jj] > Th:
                mask[ii, jj] = 1; 
            elif (nb_magn > Th).any():
                mask[ii, jj] = 1;   
        
            # noise supression
            if nb_original_sigma < noise_sigma:
                mask[ii, jj] = 0; 
                
    return mask;
