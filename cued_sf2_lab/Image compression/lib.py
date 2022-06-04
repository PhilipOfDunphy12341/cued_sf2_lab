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

def c_ratio_pca(X,quantised_bases):
    reference_scheme = bpp(quantise(X,17))*256*256
    compressed_scheme = 0
    for i in (quantised_bases[0]).T:
        compressed_scheme += bpp(i)*(np.shape(i)[0])
    for i in quantised_bases[1]:
        compressed_scheme += bpp(i)*(np.shape(i)[0])
    return (reference_scheme/compressed_scheme), compressed_scheme

from skimage.metrics import structural_similarity as ssim
def ssi(original,decoded, full):
    ssim_image = ssim(original, decoded)
    return ssim_image

def pca_encoding(images,step,max_step,cut_off,ssim = True):
    
    u,vt, percent,s = pca(images)
    
    #percent is the amount of variance that each basis captures, qunatised in proportion to this
    vt_q = []

    scale = max_step / percent[0]
    #"Equal MSE quantising"
    for i,x in enumerate(vt):
        # print(-scale*(percent[i] -percent[0])+step)
        vt_q.append(quantise(x*(255/np.amax(x)),-scale*(percent[i] -percent[0])+step)/(255/np.amax(x)))
    u_q = []
    for i,x in enumerate(u):
        u_q.append(quantise(x*(255/np.amax(x)),-scale*(percent[i] -percent[0])+step)/(255/np.amax(x)))
        
    
    dim1 = np.shape(u)[0]
    dim2 = np.shape(u)[0]
    smat = np.zeros((dim1,dim2))
    smat[:dim2, :dim2] = np.diag(s)
    
    smat_cut = smat[:cut_off,:cut_off]
    
    u_q_cut = np.array(u_q)[:,:cut_off]
    vt_q_cut = np.array(vt_q)[:cut_off,:]
    # print(u_q_cut.shape,vt_q_cut.shape,smat_cut.shape)
    q_reconstructed_image = (np.dot((u_q_cut), np.dot(smat_cut, vt_q_cut)))
    q_reconstructed_image += 128
    # reconstructed_image = (np.dot(vt.T, np.dot(smat, vt)))
    # reconstruction_error = np.std(images - reconstructed_image)
    if ssim == True:
        print(np.shape(images),np.shape(q_reconstructed_image))
        print((np.max(q_reconstructed_image)),(np.max(images)))
        reconstruction_error_q  = ssi(images,q_reconstructed_image,full = True)
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
    return np.array(u_q_cut),np.array(vt_q_cut),q_reconstructed_image,reconstruction_error_q

# from scipy import optimize
# def minf(x):
#     return x[0]**2 + (x[1]-1.)**2
# def meta_optimise_pca():
#     while 

def optimise_pca(image,min_error,ssim = True):
    step_max = 30
    step = 1
    epsilon = 1
    cut_off = 128
    min_ssim = ssi(image,quantise(image,17),full = True)
    f_k1 = pca_encoding(image,step,step_max,cut_off,ssim=ssim)[3]
    print(f_k1,min_ssim)
    if ssim == True:
        while f_k1<0.95:
            print("YYYYYYYYYYY")
            print(f_k1)
            backward_step = pca_encoding(image,step,step_max-epsilon,cut_off,ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max+epsilon,cut_off,ssim=ssim)[3]
            if backward_step > forward_step:
                new_step_max = step_max - epsilon
                f_k1 = backward_step
            elif backward_step < forward_step:
                new_step_max = step + epsilon
                f_k1 = forward_step
            step_max = new_step_max
            
        while min_ssim<f_k1:
            backward_step = pca_encoding(image,step,step_max,cut_off-epsilon,ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max,cut_off+epsilon,ssim=ssim)[3]
            if backward_step < forward_step:
                new_cut_off = cut_off - epsilon
                f_k1 = backward_step
            elif backward_step > forward_step:
                new_cut_off = cut_off + epsilon
                f_k1 = forward_step

            cut_off = int(new_cut_off)
    else:
        while f_k1>min_error:
            backward_step = pca_encoding(image,step,step_max-epsilon,cut_off,ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max+epsilon,cut_off,ssim=ssim)[3]
            if backward_step < forward_step:
                new_step_max = step_max - epsilon
                f_k1 = backward_step
            elif backward_step > forward_step:
                new_step_max = step + epsilon
                f_k1 = forward_step
            step_max = new_step_max
            
        while 4.86>f_k1:
            backward_step = pca_encoding(image,step,step_max,cut_off-epsilon,ssim=ssim)[3]
            forward_step = pca_encoding(image,step,step_max,cut_off+epsilon,ssim=ssim)[3]
            if backward_step > forward_step:
                new_cut_off = cut_off - epsilon
                f_k1 = backward_step
            elif backward_step < forward_step:
                new_cut_off = cut_off + epsilon
                f_k1 = forward_step

            cut_off = int(new_cut_off)
        
    return f_k1,step_max,cut_off

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
