import jpegNew as jp
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
from familiarisation import load_mat_img, plot_image
from skimage.metrics import structural_similarity as ssim

# Create a dcitionary of test images and the given miages to it
TestImages= {}

TestImages['Lighthouse'] = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]
TestImages['Bridge'] = load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})[0]
TestImages['Flamingo'] = load_mat_img(img='flamingo.mat', img_info='X', cmap_info={})[0]

# Create dictionary which will contain the images, SSI, and rms erros for each optimisation algorithm
CompSchemeData = {}

#Contains the reconstructured image, rms and SSI for each test image
JpegConstStepData = {}

for i in TestImages:
        #print(X)
        #XZero = np.subtract(TestImages[i], 128)
        #print(XZero)
        test = jp.JpegOptimiseConstStep(TestImages[i], 100, 0.5, 100)

        #print(test[1:])

        Z = jp.jpegdec(vlc = test[0][0], qstep = test[-1], bits = test[0][1], huffval = test[0][2])

        rms = np.std(TestImages[i] - Z)
        ssm, fullIm = ssim(TestImages[i], Z, full = True)
        #print(ssm)
        fig, ax = plt.subplots()
        plot_image(fullIm, ax=ax)
        plt.show()
        JpegConstStepData[i] = (Z, rms, ssm)

print(JpegConstStepData)

"""fig = plt.figure()
for i in JpegConstStepData:
        fig.add_subplot(i[0])
        plt.imshow(i[0])

plt.show()"""