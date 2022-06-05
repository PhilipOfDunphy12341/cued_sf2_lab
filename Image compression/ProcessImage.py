#It should print out the number of bits to code each image and the rms error between the coded 
#image and the original, and display the original and the coded image side by side.


#40,960bits maximum encoded size  

#DEMONSTRATION OF PCA COMPRESSION ON lighthouse
#Minimises quantisation parameters (Equal MSE) until error is that of pure quantising with 17, returns compression ratios and error

from cued_sf2_lab.familiarisation import load_mat_img
from lib import *

X_pre_zero_mean, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map'})
X = X_pre_zero_mean

_, new_step_max ,new_cut_off= optimise_pca(X,3,ssim = True)
q_u,q_bases,q_reconstructed_image,reconstruction_error_q, smat_cut,variance ,q_u_dct,q_bases_dct,q_reconstructed_image_dct,reconstruction_error_q_dct= pca_encoding(X,1,new_step_max,new_cut_off,ssim = True)
print(np.shape(q_bases))
print(np.shape(q_bases))
# plot_all_images(q_u)
# plot_all_images(q_bases)
fig, ax = plt.subplots(nrows = 1, ncols=2)

ax[0].imshow(q_reconstructed_image,cmap = 'gray')
ax[1].imshow(quantise(X,17),cmap = 'gray')

plt.show()  

fig, ax = plt.subplots(nrows = 1, ncols=2)

ax[0].imshow(q_reconstructed_image_dct,cmap = 'gray')
ax[1].imshow(quantise(X,17),cmap = 'gray')

plt.show()  
ratio, bits_encoding, ratio_dct = c_ratio_pca(X,(q_u,q_bases),smat_cut,(q_u_dct,q_bases_dct))
print("Compression ratio: ",ratio, " Bits: ", bits_encoding, " DCT: ",ratio_dct)
#Search method is currently inefficient finding first the optimal quantisation level to some arbitrary level and then discarding principal 
#components until the error is satisfied, upgrading to a 2D search would be better.

#Currently, we need to store both quantised eigen vector matricies (U and V.T) and im not sure if there is a better way to do this
#with linear algebra magic given our image is square (in the 3G3 lab they only care about the latter so not sure if im missing something)

#It seems like if you want to focus on compression, then prioritise discarding PCs instead of quantising, if you want better quality then
#dont let the "Max quantisation level at smalest singualar value" get too small as too many PCs will need to be discarded to get a good CR

#Next step is to use SSI to optimise instead

# print("PHASE 2")
# print(block_pca(X))