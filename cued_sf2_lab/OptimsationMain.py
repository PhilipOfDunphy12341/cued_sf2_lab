import JpegNew as jp
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
from familiarisation import load_mat_img, plot_image
from skimage.metrics import structural_similarity as ssim
from dwt import dwtQuantMatrix, dwtImpulseResponse

# Create a dictionary, the key being the image name and the value being the image
Images= {}

# Add images to dictionary

Images['Lighthouse'] = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})[0]
#Images['Bridge'] = load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})[0]
#Images['Flamingo'] = load_mat_img(img='flamingo.mat', img_info='X', cmap_info={})[0]
#Images['2018'] = load_mat_img(img='SF2_competition_image_2018.mat', img_info='X', cmap_info={})[0]
#Images['2019'] = load_mat_img(img='SF2_competition_image_2019.mat', img_info='X', cmap_info={})[0]
#Images['2020'] = load_mat_img(img='SF2_competition_image_2020.mat', img_info='X', cmap_info={})[0]
#Images['2021'] = load_mat_img(img='SF2_competition_image_2021.mat', img_info='X', cmap_info={})[0]
#Images['2022'] = load_mat_img(img='SF2_Competition_Image2022.mat', img_info='X', cmap_info={})[0]

# List for Rms and structural similarity values
RmsList = []
SSMList = []

# Loop for DCT
for i in Images:
        print("New image")
        X = Images[i].copy()
        X = X - 128.0

        DCTOptimal = jp.JpegOptimiseConstStep(X, InitStep = 75, Increment = 1, MaxError = 50)
        Z = jp.jpegdec(vlc = DCTOptimal[0][0], qstep = DCTOptimal[-1], bits = DCTOptimal[0][1], huffval = DCTOptimal[0][2])

        rms = np.std(X - Z)
        RmsList.append(rms)
        ssm, fullIm = ssim(X, Z, full = True)
        SSMList.append(ssm)
        print("Rms is: ", rms)
        print("SSm is: ", ssm)

        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)
        plt.savefig('DCT.png')
        plt.show()

        
#Loop for constant step size dwt
for i in Images:
        print("New image")
        X = Images[i].copy()
        X = X - 128.0

        # Finds optimal step
        DWTConstStep = jp.dwtOptimiseConstStep(X, InitStep = 75, Increment = 3, MaxError = 50, levels = 4, dcbits = 8)
        optimal_dwtmat = dwtQuantMatrix(qstep = DWTConstStep[-1], levels = 4, EqualStep= True)
        Z = jp.dwtdec(vlc = DWTConstStep[0][0], dwtqstep = optimal_dwtmat, bits = DWTConstStep[0][1], huffval = DWTConstStep[0][2], levels = 4)

        rms = np.std(X - Z)
        RmsList.append(rms)
        ssm, fullIm = ssim(X, Z, full = True)
        SSMList.append(ssm)
        print("Rms is: ", rms)
        print("SSm is: ", ssm)

        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)
        plt.savefig('ConstStepDWT.png')
        plt.show()

#Loop for constant mse dwt
for i in Images:
        X =Images[i].copy()
        X = X - 128.0

        test = jp.dwtOptimiseEqualMse(X, InitStep = 75, Increment = 1, MaxError = 50, levels = 3, dcbits = 8)
        energy_out = dwtImpulseResponse(levels = 3)
        optimal_dwtmat = dwtQuantMatrix(qstep = test[-1], levels = 3, energies = energy_out, EqualStep= False, EqualMse= True)
        Z = jp.dwtdec(vlc = test[0][0], dwtqstep = optimal_dwtmat, bits = test[0][1], huffval = test[0][2], levels = 3)

        rms = np.std(X - Z)
        RmsList.append(rms)
        ssm, fullIm = ssim(X, Z, full = True)
        SSMList.append(ssm)
        print("Rms is: ", rms)
        print("SSm is: ", ssm)

        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)
        plt.savefig('ConstMseDWT.png')
        plt.show()


