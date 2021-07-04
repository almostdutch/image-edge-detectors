"""
demo_edge_detectors_utils.py

four edge detectors:
    
    (1) Sobel edge detector
        Strong response for horizontal and vertial edges
        
    (2) Laplacian edge detector:
        Strong response for horizontal, vertial and diagonal edges
        
    (3) Single line directional edge detector:
        Strong response for single line horizontal, vertial or diagonal edges
    
    (4) Canny edge detector:
        Strong response for horizontal, vertial and diagonal edges

"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
from edge_detectors_utils import SobelEdgeDetector, LaplacianEdgeDetector, \
    SingleLineEdgeDetector, CannyEdgeDetector

# load test image
image_full = np.array(mpimg.imread('sample_image.tif'));
image_full = image_full.astype(np.float64);

# add independent Gaussian noise
N1, N2 = image_full.shape;
mu = 0; # mean value
sigma = 10; # standard deviation
noise = np.random.normal(mu, sigma, size = (N1, N2));
image_full_noise = image_full + noise; 

# show images (original and degraded)
fig_width, fig_height = 5, 5;
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image_full, cmap='gray')
ax1.set_title("image original")
ax1.set_axis_off()

ax2.imshow(image_full_noise, cmap='gray')
ax2.set_title("image + noise")
ax2.set_axis_off()
plt.tight_layout()


# Sobel edge detector
kernel_sigma = 1; # Gaussian kernel standard deviation for denoising
threshold = 0.3; # [0, 1]
noise_sigma = 10; # noise standard deviation
mask_sobel = SobelEdgeDetector(image_full_noise, kernel_sigma, threshold, noise_sigma);

# Laplacian edge detector
kernel_sigma = 1; # Gaussian kernel standard deviation for denoising
threshold = 0.2; # [0, 1]
noise_sigma = 10; # noise standard deviation
mask_laplacian = LaplacianEdgeDetector(image_full_noise, kernel_sigma, threshold, noise_sigma);

# Single line directional edge detector
orientation = 'ver'; # 'hor' (horizontal), 'ver' (vertical), \
#   '45deg' (plus 45 degrees), and 'm45deg' (minus 45 degrees)
kernel_sigma = 1; # Gaussian kernel standard deviation for denoising
threshold = 0.1; # [0, 1]
noise_sigma = 10; # noise standard deviation
mask_directional = SingleLineEdgeDetector(image_full_noise, kernel_sigma, threshold, noise_sigma, orientation);

# Canny edge detector
kernel_sigma = 1; # Gaussian kernel standard deviation for denoising
threshold = [0.05, 0.3]; # [0, 1]
noise_sigma = 10; # noise standard deviation
mask_canny = CannyEdgeDetector(image_full_noise, kernel_sigma, threshold, noise_sigma);

# show images (original and degraded)
fig_width, fig_height = 5, 5;
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(mask_sobel, cmap='gray')
ax1.set_title("sobel \n edge detector")
ax1.set_axis_off()

ax2.imshow(mask_laplacian, cmap='gray')
ax2.set_title("laplacian \n edge detector")
ax2.set_axis_off()

ax3.imshow(mask_directional, cmap='gray')
ax3.set_title(f'directional ({orientation}) \n edge detector')
ax3.set_axis_off()

ax4.imshow(mask_canny, cmap='gray')
ax4.set_title("canny \n edge detector")
ax4.set_axis_off()
plt.tight_layout()