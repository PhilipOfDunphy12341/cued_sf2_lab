#It should print out the number of bits to code each image and the rms error between the coded 
#image and the original, and display the original and the coded image side by side.


#40,960bits maximum encoded size  

#DEMONSTRATION OF PCA COMPRESSION ON lighthouse
#Minimises quantisation parameters (Equal MSE) until error is that of pure quantising with 17, returns compression ratios and error

from cued_sf2_lab.familiarisation import load_mat_img
from lib import *
import numpy as np

X_pre_zero_mean, cmaps_dict = load_mat_img(img='SF2_Competition_Image2022.mat', img_info='X', cmap_info={})
X = X_pre_zero_mean

_, new_step_max ,new_cut_off= optimise_pca(X,3,ssim = True)
q_u,q_bases,q_reconstructed_image,reconstruction_error_q, smat_cut,variance ,q_u_dct,q_bases_dct,q_reconstructed_image_dct,reconstruction_error_q_dct,bits,dct_bits= pca_encoding(X,1,new_step_max,new_cut_off,ssim = True)
print(np.shape(q_bases))
print(np.shape(q_bases))
# plot_all_images(q_u)
# plot_all_images(q_bases)
fig, ax = plt.subplots(nrows = 1, ncols=3)

ax[0].imshow(q_reconstructed_image,cmap = 'gray')
ax[1].imshow(quantise(X,17),cmap = 'gray')
ax[2].imshow(q_reconstructed_image_dct,cmap = 'gray')

plt.show() 
import cv2
import numpy as np

img = q_reconstructed_image_dct # Your image as a numpy array 

cv2.imwrite("PCA-DCT_2022_2.png", img)



ratio, bits_encoding, ratio_dct , compressed_scheme_dct= c_ratio_pca(X,(q_u,q_bases),smat_cut,(q_u_dct,q_bases_dct))
print("Compression ratio: ",ratio, " Bits: ", bits_encoding, " PCA/DCT ratio: ",ratio_dct, " PCA/DCT bits: ",compressed_scheme_dct)
#Search method is currently inefficient finding first the optimal quantisation level to some arbitrary level and then discarding principal 
#components until the error is satisfied, upgrading to a 2D search would be better.

#Currently, we need to store both quantised eigen vector matricies (U and V.T) and im not sure if there is a better way to do this
#with linear algebra magic given our image is square (in the 3G3 lab they only care about the latter so not sure if im missing something)

#It seems like if you want to focus on compression, then prioritise discarding PCs instead of quantising, if you want better quality then
#dont let the "Max quantisation level at smalest singualar value" get too small as too many PCs will need to be discarded to get a good CR

#Next step is to use SSI to optimise instead

# print("PHASE 2")
# print(block_pca(X))

####  We now want to huffman encode our DCT quantised U,VT and our original S directly



print(np.shape(q_u_dct))
print(np.shape(q_bases_dct))
# print(np.shape(q_u_dct))
# print(q_u_dct)
# print("\n")
# print(q_bases_dct)
# print("\n")
# print(smat_cut)
print("ASDASDASDASDASD")

minx = np.min(q_u_dct, axis=None)
maxx = np.max(q_u_dct, axis=None)
# Calculate histogram of x in bins defined by bins.
bins = list(range(int(np.floor(minx)), int(np.ceil(maxx)+1)))


h, s = np.histogram(q_u_dct, bins)
# Convert bin counts to probabilities, and remove zeros.
p = h / np.sum(h)
p = p[p > 0]
plt.plot(h)
plt.show()
minx = np.min(q_bases_dct, axis=None)
maxx = np.max(q_bases_dct, axis=None)
# Calculate histogram of x in bins defined by bins.
bins = list(range(int(np.floor(minx)), int(np.ceil(maxx)+1)))


h, s = np.histogram(q_bases_dct, bins)
# Convert bin counts to probabilities, and remove zeros.
p = h / np.sum(h)
p = p[p > 0]
plt.plot(h)
plt.show()
minx = np.min(smat_cut, axis=None)
maxx = np.max(smat_cut, axis=None)
# Calculate histogram of x in bins defined by bins.
bins = list(range(int(np.floor(minx)), int(np.ceil(maxx)+1)))


h, s = np.histogram(smat_cut, bins)
# Convert bin counts to probabilities, and remove zeros.
p = h / np.sum(h)
p = p[p > 0]
plt.plot(h)
plt.show()

data_in = [np.ravel(q_bases_dct),np.ravel(q_u_dct),np.ravel(smat_cut)]
encoded = []
code_books = []
from trees import *
for store in data_in:
    probabilities = {}
    for i in store:
        # print(i)
        if i in probabilities:
            probabilities[i] += 1
        else:
            probabilities[i] = 1
    total = np.sum(list(probabilities.values()))

    for i in probabilities.keys():
        probabilities[i] = probabilities[i] /total
    # print(probabilities)

    def huffman(p):

        xt = [[-1,[], a] for a in p]

        p = [(k,p[a]) for k,a in zip(range(len(p)),p)]

        nodelabel = len(p)

        while len(p) > 1:

            p = sorted(p,key = lambda el:el[1])

            xt.append([-1,[],str(nodelabel)])

            nodelabel += 1

            xt[p[0][0]][0] = len(xt)-1
            xt[p[1][0]][0] = len(xt)-1

            xt[-1][1] = [p[0][0],p[1][0]]

            p.append((len(xt)-1,p[0][1] + p[1][1]))

            p.pop(0)
            p.pop(0)

        return(xt)

    def vl_encode(x, c):
        y = []
        for a in x:
            y.extend(c[a])
        return y

    def vl_decode(y, xt):
        x = []
        root = [k for k in range(len(xt)) if xt[k][0]==-1]
        if len(root) != 1:
            raise NameError('Tree with no or multiple roots!')
        root = root[0]
        leaves = [k for k in range(len(xt)) if len(xt[k][1]) == 0]

        n = root
        for k in y:
            if len(xt[n][1]) < k:
                raise NameError('Symbol exceeds alphabet size in tree node')
            if xt[n][1][k] == -1:
                raise NameError('Symbol not assigned in tree node')
            n = xt[n][1][k]
            if len(xt[n][1]) == 0: # it's a leaf!
                x.append(xt[n][2])
                n = root
        return x



    
    xt = huffman(probabilities)

    # print(xtree2newick(xt))

    c = xtree2code(xt)
    hamlet_huf = vl_encode(store, c)

    code_books.append(xt)
    print(len(hamlet_huf))
    encoded.append(hamlet_huf)
# print(hamlet_huf)

decoded = []
for i,encoded_store in enumerate(encoded):

#To decode, we need to provide the huffman table as well as the number of bins removed so the data stream can be partitioned back into its correct 3 components
    hamlet_decoded = vl_decode(encoded_store, code_books[i])
    decoded.append(hamlet_decoded)
    print(len(hamlet_decoded))
    print("BBBBBBBBBB")
    print(np.sum((hamlet_decoded) - data_in[i]))

# print(hamlet_decoded)

dec_q_u_dct = np.reshape(decoded[1],(21,256))
dec_vt_u_dct = np.reshape(decoded[0],(21,256))
dec_s = decoded[2]


from cued_sf2_lab.dct import dct_ii, dct_iv, colxfm, regroup

max_value = new_step_max/variance[0]
C8 = dct_ii(8)

u_q_cut_recovered = np.array([np.ravel(colxfm(colxfm(np.reshape(Yq,(16,16)).T, C8.T).T, C8.T)) for Yq in dec_q_u_dct]).T
vt_q_cut_recovered = np.array([np.ravel(colxfm(colxfm(np.reshape(Yq,(16,16)).T, C8.T).T, C8.T)) for Yq in dec_vt_u_dct])

print(np.shape(u_q_cut_recovered))
print(np.shape(vt_q_cut_recovered))

print(np.shape(dec_s))

jeff =  np.dot(  np.diag(dec_s), (vt_q_cut_recovered/(128/max_value))  ) 
print(np.shape(jeff))

q_reconstructed_image_dct = ( np.dot(  (u_q_cut_recovered/(128/max_value))   , jeff ))
q_reconstructed_image_dct += 128

plt.imshow(q_reconstructed_image_dct)
plt.show()