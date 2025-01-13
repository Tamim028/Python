
"""
NLM - Skimage
NLM - opencv
BM3D Block-matching and 3D filtering,
Bilateral, 

Above 3 are the top denoising algorithms for MRI.

Gaussian,
Total variation filter, 
Wavelet denoising filter,
Shift invariant wavelet,
Markov random field

Total variation (TV) also works great. 
Bilateral is slow and it probably works fine except it takes too much 
time to experiment with parameters.
"""


import matplotlib.pyplot as plt




#Gaussian
from skimage import data, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, cycle_spin, denoise_nl_means)
from skimage.util import random_noise
import cv2
import numpy as np


noisy_img = img_as_float(io.imread("DenoisingImages/MRI_Images/MRI_noisy.tif", as_gray=True))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("DenoisingImages/MRI_Images/MRI_clean.tif", as_gray=True))


# Gaussian #                    
gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)
plt.imshow(gaussian_img, cmap='gray')
plt.imsave("DenoisingImages/MRI_Images/Gaussian_smoothed.tiff", gaussian_img, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)


######################################################################
# Bilateral, TV and Wavelet


sigma_est = estimate_sigma(noisy_img, multichannel=True, average_sigmas=True)

denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=5,
                multichannel=False)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)
plt.imsave("DenoisingImages/MRI_Images/bilateral_smoothed.tiff", denoise_bilateral, cmap='gray')


# TV #
denoise_TV = denoise_tv_chambolle(noisy_img, weight=0.3, multichannel=False)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_TV)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", TV_cleaned_psnr)
plt.imsave("DenoisingImages/MRI_Images/TV_smoothed.tiff", denoise_TV, cmap='gray')


# Wavelet #
wavelet_smoothed = denoise_wavelet(noisy_img, multichannel=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)

plt.imsave("DenoisingImages/MRI_Images/wavelet_smoothed.tiff", wavelet_smoothed, cmap='gray')


# #####################
#Shift invariant wavelet denoising


denoise_kwargs = dict(multichannel=False, wavelet='db1', method='BayesShrink',
                      rescale_sigma=True)

all_psnr = []
max_shifts = 3     #0, 1, 3, 5

Shft_inv_wavelet = cycle_spin(noisy_img, func=denoise_wavelet, max_shifts = max_shifts,
                            func_kw=denoise_kwargs, multichannel=False)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
shft_cleaned_psnr = peak_signal_noise_ratio(ref_img, Shft_inv_wavelet)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", shft_cleaned_psnr)

plt.imsave("DenoisingImages/MRI_Images/Shift_Inv_wavelet_smoothed.tiff", Shft_inv_wavelet, cmap='gray')




# ##########################################################################
#NLM from SKIMAGE
sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))


NLM_skimg_denoise_img = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=9, patch_distance=5, multichannel=False)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
NLM_skimg_cleaned_psnr = peak_signal_noise_ratio(ref_img, NLM_skimg_denoise_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", NLM_skimg_cleaned_psnr)


denoise_img_as_8byte = img_as_ubyte(NLM_skimg_denoise_img)
plt.imsave("DenoisingImages/MRI_Images/NLM_skimage_denoised.tiff", denoise_img_as_8byte, cmap='gray')


# ###########################################################################
# #NLM opencv

# import numpy as np
# from matplotlib import pyplot as plt
# from skimage import img_as_ubyte, img_as_float
# from matplotlib import pyplot as plt
# from skimage import io
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio

# noisy_img = io.imread("images/MRI_images/MRI_noisy.tif", as_gray=True)  #Only 8 bit supported for CV2 NLM
# ref_img = io.imread("images/MRI_images/MRI_clean.tif")

# # fastNlMeansDenoising(InputArray src, OutputArray dst, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

# NLM_CV2_denoise_img = cv2.fastNlMeansDenoising(noisy_img, None, 3, 7, 21)


# plt.imsave("images/MRI_images/NLM_CV2_denoised.tif", NLM_CV2_denoise_img, cmap='gray')
# plt.imshow("images/MRI_images/NLM_CV2_denoised.tif", NLM_CV2_denoise_img, cmap='gray')


# ###########################################################################
# #BM3D Block-matching and 3D filtering
# #pip install bm3d

# import matplotlib.pyplot as plt
# from skimage import io, img_as_float
# from skimage.metrics import peak_signal_noise_ratio
# import bm3d
# import numpy as np

# noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif", as_gray=True))
# ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif"))


# BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
# #BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

#  #Also try stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING                     


# noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
# BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)
# print("PSNR of input noisy image = ", noise_psnr)
# print("PSNR of cleaned image = ", BM3D_cleaned_psnr)


# plt.imshow(BM3D_denoised_image, cmap='gray')
# plt.imsave("images/MRI_images/BM3D_denoised.tif", BM3D_denoised_image, cmap='gray')
