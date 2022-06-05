from bz2 import compress
from typing import final
import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.dct import dct_ii, dct_iv, colxfm, regroup
from cued_sf2_lab.laplacian_pyramid import quantise, bpp

def dwt_encoding():
    #Should take an image,
    pass
def dwt_decoding():
    pass

def dct_encoding():
    
    
    pass
def dct_decoding():
    #Should take a compressed image, the DCT size and return the original image
    pass


def sample_patches(imgs, sz, n_sub):
    """
    Sample image patches
    
    Parameters
    ----------
    imgs
        images with shape (N_img, N_pix)
    sz 
        side-length (in pixels) of the image patches
    n_sub
        number of image patches to sample
    
    Returns
    -------
    S 
        matrix of image patches with shape (n_sub, n_pix), where n_pix =  sz * sz
    """
    N_img = imgs.shape[0]
    samples = []
    osz = int(np.sqrt(len(imgs[0])))
    for _ in range(n_sub):
        img = imgs[np.random.randint(N_img)].reshape((osz, osz))
        ii = np.random.randint(osz - sz + 1)
        jj = np.random.randint(osz - sz + 1)
        c = img[ii:ii + sz, jj:jj + sz].reshape(1, -1)
        samples.append(c)
    return np.concatenate(samples, 0)


def pca(x):
    """
    PCA
    ---
    
    Parameters
    ----------
    x: matrix of subimages with shape (n_sub, n_pix)
    
    Returns
    -------
    bases: basis functions with shape (n_pix, n_pix)
    pct: percentage of variance captured by each basis functions (each row of bases)
    """
    x = x - np.mean(x)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    var = np.square(s)
    pct = var / np.sum(var)
    return u,vt, pct, s

def c_ratio_pca(X,quantised_bases,s,dct_quantised_bases):
    reference_scheme = bpp(quantise(X,17))*256*256
    compressed_scheme = 0
    compressed_scheme_dct = 0
    for i in (quantised_bases[0]).T:
        compressed_scheme += bpp(i)*(np.shape(i)[0])
    for i in quantised_bases[1]:
        compressed_scheme += bpp(i)*(np.shape(i)[0])
    for i in s:
        compressed_scheme += np.log2(i)
    
    for i in dct_quantised_bases[0]:
        compressed_scheme_dct+= bpp(i)*(np.shape(i)[0])
    for i in dct_quantised_bases[1]:
        compressed_scheme_dct+= bpp(i)*(np.shape(i)[0])
    for i in s:
        compressed_scheme_dct += np.log2(i)
        
    return (reference_scheme/compressed_scheme), compressed_scheme,(reference_scheme/compressed_scheme_dct),compressed_scheme_dct

from skimage.metrics import structural_similarity as ssim
def ssi(original,decoded, full):
    ssim_image = ssim(original, decoded)
    return ssim_image

def pca_encoding(images,step,max_step,cut_off,ssim = True):
    N=8
    u,vt, percent,s = pca(images)
    # print(np.shape(u))
    # print(np.shape(vt))
    C8 = dct_ii(N)
    #percent is the amount of variance that each basis captures, qunatised in proportion to this
    vt_q = []
    vt_q_dct = []
    scale = max_step / percent[0]
    #"Equal MSE quantising"
    
    if np.max(vt)>np.max(u):
        max_value= np.max(vt)
    else:
        max_value = np.max(u)
        
        
    for i,x in enumerate(vt):
        # print(-scale*(percent[i] -percent[0])+step)
        quantise_standard = -scale*(percent[i] -percent[0])+step
        x_16 = np.reshape(x*(128/max_value),(16,16))
        
        #Now do dct on this basis function
        y = colxfm(colxfm(x_16, C8).T, C8).T 

        vt_q.append(quantise(x*(128/max_value),quantise_standard))
        vt_q_dct.append((quantise(np.ravel(y),quantise_standard)))
        
        y_q =quantise(y,quantise_standard)
        
        # fig,ax = plt.subplots(nrows = 1,ncols =2)
        # ax[0].imshow(np.reshape(x,(16,16)))
        # ax[1].imshow(np.reshape(colxfm(colxfm(np.reshape(y_q,(16,16)).T, C8.T).T, C8.T),(16,16)))
        # plt.show()

    u_q = []
    u_q_dct = []

    for i,x in enumerate(u):
        quantise_standard = -scale*(percent[i] -percent[0])+step
        
        
        u_q.append(quantise(x*(128/max_value),quantise_standard))
        
    for i,x in enumerate(u.T):
        quantise_standard = -scale*(percent[i] -percent[0])+step
        x_16 = np.reshape(x*(128/max_value),(16,16))
        y = colxfm(colxfm(x_16, C8).T, C8).T 
        u_q_dct.append((quantise(np.ravel(y),quantise_standard)))

        # fig,ax = plt.subplots(nrows = 1,ncols =2)
        # ax[0].imshow(np.reshape(x,(16,16)))
        # ax[1].imshow(np.reshape(colxfm(colxfm(np.reshape(y_q,(16,16)).T, C8.T).T, C8.T),(16,16)))
        # plt.show()
        
    dim1 = np.shape(u)[0]
    dim2 = np.shape(u)[0]
    smat = np.zeros((dim1,dim2))
    
    smat[:dim2, :dim2] = np.diag(s)
    
    smat_cut = smat[:cut_off,:cut_off]
    
    # print(smat_cut)
    u_q_cut = np.array(u_q)[:,:cut_off]
    vt_q_cut = np.array(vt_q)[:cut_off,:]
    
    
 
    
    u_q_cut_dct = np.array(u_q_dct)[:cut_off,:]
    vt_q_cut_dct = np.array(vt_q_dct)[:cut_off,:]
    
    # minx = np.min(u_q_cut_dct, axis=None)
    # print(minx)
    # maxx = np.max(u_q_cut_dct, axis=None)
    # bins = list(range(int(np.floor(minx)+1), int(np.ceil(maxx)+1)))
    # print(maxx)
    # print(bins)

    # h, s = np.histogram(np.ravel(u_q_cut_dct), bins)
    # # Convert bin counts to probabilities, and remove zeros.
    # p = h / np.sum(h)
    # p = p[p > 0]
    # plt.plot(h)
    # plt.show()
    # print(np.shape(u_q_cut))
    # print(np.shape(vt_q_cut))
    # print(np.shape(u_q_cut_dct))
    # print(np.shape(vt_q_cut_dct))

    # print(vt_q_cut[0])
    # print(vt_q_cut[0]/(128/max_value))
    # print(np.max(vt_q_cut)-np.min(vt_q_cut),np.max(vt_q_cut)-np.min(vt_q_cut/(128/max_value)))
    
    
    
    #DECODING#
    
    # print(u_q_cut.shape,vt_q_cut.shape,smat_cut.shape)
    q_reconstructed_image = (np.dot((u_q_cut/(128/max_value)), np.dot(smat_cut, (vt_q_cut/(128/max_value)))))
    q_reconstructed_image += 128
    
    u_q_cut_recovered = np.array([np.ravel(colxfm(colxfm(np.reshape(Yq,(16,16)).T, C8.T).T, C8.T)) for Yq in u_q_cut_dct]).T
    vt_q_cut_recovered = np.array([np.ravel(colxfm(colxfm(np.reshape(Yq,(16,16)).T, C8.T).T, C8.T)) for Yq in vt_q_cut_dct])
    # plt.imshow(colxfm(colxfm(np.reshape(vt_q_cut_dct[0],(16,16)).T, C8.T).T, C8.T))
    # plt.show()
    # print(np.shape(u_q_cut_recovered))
    # print(np.shape(vt_q_cut_recovered))
    
    q_reconstructed_image_dct = (np.dot((u_q_cut_recovered/(128/max_value)), np.dot(smat_cut, (vt_q_cut_recovered/(128/max_value)))))
    q_reconstructed_image_dct += 128
    # reconstructed_image = (np.dot(vt.T, np.dot(smat, vt)))
    # reconstruction_error = np.std(images - reconstructed_image)
    if ssim == True:
        reconstruction_error_q  = ssi(images,q_reconstructed_image,full = True)
        reconstruction_error_q_dct  = ssi(images,q_reconstructed_image_dct,full = True)
        # print("TTTTT")
        # print(reconstruction_error_q,reconstruction_error_q_dct)

    elif ssim == False:
        reconstruction_error_q = np.std(images-q_reconstructed_image)
    # print(reconstruction_error)
    print("Reconstruction error: ",round(reconstruction_error_q,2)," Max quantisation level at smallest singular value: ", max_step, "No. of singular values retained: ",cut_off)
    # plt.imshow(q_reconstructed_image, cmap = 'gray')
    # plt.show()
    # print(images[1,:100])
    # print(q_reconstructed_image[1,:100])
    # plt.imshow(images, cmap = 'gray')
    # plt.show()
    # plt.imshow(full_im, cmap = 'gray')
    # plt.show()
    _,bits,_,_ = c_ratio_pca(images,(u_q_cut,vt_q_cut),s[:cut_off],(u_q_cut_dct,vt_q_cut_dct))
    # print(bits)
    return np.array(u_q_cut),np.array(vt_q_cut),q_reconstructed_image,reconstruction_error_q, s[:cut_off],percent, u_q_cut_dct,vt_q_cut_dct,q_reconstructed_image_dct,reconstruction_error_q_dct,bits

def dctbpp(Yr, N):
    Yr = regroup(Yr,8)

    total_bits = 0
    l = int(16/2)
    for i in np.arange(0,16,l):

        
        for j in np.arange(0,16,l):
            subimage = Yr[i:i+l,j:j+l]
            bits_sub = bpp(subimage)*l*l
            total_bits += bits_sub
            # Yq_s[i:i+l,j:j+l] = quantise(subimage,17)
    return total_bits#, Yq_s

def optimise_pca(image,min_error,ssim = True):
    step_max = 30
    step = 1
    epsilon = 1
    epsilon_q = 1
    cut_off = 128
    # min_ssim = ssi(image,quantise(image,17),full = True)
    f_k1 = pca_encoding(image,step,step_max,cut_off,ssim=ssim)[10]
    if ssim == True:
        new_cut_off = cut_off
        while f_k1>150000:
            print(f_k1)

            backward_step = pca_encoding(image,step,step_max-epsilon_q,cut_off,ssim=ssim)[10]
            forward_step = pca_encoding(image,step,step_max+epsilon_q,cut_off,ssim=ssim)[10]
            print(backward_step,forward_step)
            if backward_step > forward_step:
                new_step_max = step_max - epsilon_q
                f_k1 = backward_step
            elif backward_step < forward_step:
                new_step_max = step + epsilon_q
                f_k1 = forward_step
            step_max = new_step_max
            
        while 40960<f_k1:
            print(f_k1)

            backward_step = pca_encoding(image,step,step_max,(cut_off-epsilon),ssim=ssim)[10]
            forward_step = pca_encoding(image,step,step_max,(cut_off+epsilon),ssim=ssim)[10]
            if backward_step < forward_step:
                new_cut_off = cut_off - epsilon
                f_k1 = backward_step
            elif backward_step > forward_step:
                new_cut_off = cut_off + epsilon
                f_k1 = forward_step

            cut_off = (new_cut_off)
    else:
        while f_k1>min_error:
            backward_step = pca_encoding(image,step,step_max-epsilon_q,cut_off,ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max+epsilon_q,cut_off,ssim=ssim)[3]
            if backward_step < forward_step:
                new_step_max = step_max - epsilon_q
                f_k1 = backward_step
            elif backward_step > forward_step:
                new_step_max = step + epsilon_q
                f_k1 = forward_step
            step_max = new_step_max
            
        while 4.86>f_k1:
            backward_step = pca_encoding(image,step,step_max,(cut_off-epsilon),ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max,(cut_off+epsilon),ssim=ssim)[3]
            if backward_step > forward_step:
                new_cut_off = cut_off - epsilon
                f_k1 = backward_step
            elif backward_step < forward_step:
                new_cut_off = cut_off + epsilon
                f_k1 = forward_step

            cut_off = (new_cut_off)
        
    return f_k1,step_max,cut_off


        
def block_pca(original_image,block_size=32):
    subimages = []

    for i in range(0, original_image.shape[0], block_size):  

        for j in range(0, original_image.shape[1], block_size):
            print(i,j)
            subimages.append(original_image[i:i+block_size, j:j+block_size])
    q_u_list = []
    q_bases_list = []
    smat_cut_list = []
    final_image=[]
    for subimage in subimages:
        _, new_step_max ,new_cut_off= optimise_pca(subimage,3,ssim = True)
        q_u,q_bases,q_reconstructed_image,reconstruction_error_q, smat_cut,variance = pca_encoding(subimage,1,new_step_max,new_cut_off,ssim = True)
        q_u_list.append(q_u)
        q_bases_list.append(q_bases)
        smat_cut_list.append(smat_cut)
        final_image.extend(reconstruction_error_q)
    final_image = np.reshape(final_image,(256,256))
    reconstruction_error_q_final = ssi(original_image,final_image)
    plt.imshow(final_image,cmap = 'gray')
    plt.show()
    return reconstruction_error_q_final

def plot_image(x, i):
    """
    Parameters
    ----------
    x
        images (e.g. imgs["I2"])
    i
        ith image of x (i.e., x[i])

    Returns
    -------
    None
    """
    if i > x.shape[0] - 1:
        raise Exception("index %i out of bounds, total number of images: %i" %
                        (i, x.shape[0]))
    n_pix = x.shape[1]
    x = x[i]
    sz = int(np.sqrt(n_pix))
    x = x.reshape(sz, sz)
    plt.figure()
    plt.imshow(x, interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_all_images(x):
    """
    Parameters
    ----------
    x 
        images with shape (n_img, n_pix)
        e.g., imgs["I2"]

    Returns
    -------
    None
    """
    n_img, n_pix = x.shape
    if n_img > 1:
        n_cols = int(np.ceil(np.sqrt(n_img)))
        n_rows = n_cols
        _, axes = plt.subplots(n_rows,
                               n_cols,
                               figsize=(10, 10),
                               sharex=True,
                               sharey=True)
        sz = int(np.sqrt(n_pix))
        for i in range(n_img):
            acol = i % n_cols
            arow = (i - acol) // n_cols
            axes[arow, acol].imshow(x[i].reshape((sz, sz)), interpolation="nearest")
            axes[arow, acol].axis("off")
    else:
        plot_image(x, 0)
    plt.show()



def pca_decoding():
    pass

def huffman_encoding():
    pass
def huffman_decoding():
    pass
