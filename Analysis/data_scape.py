from astropy.io import fits
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import seaborn as sn
import pickle

os.chdir("../")

from Data_handling.spectra_object import Spectrum, learning_data
from Data_handling.pca_functions import Pca_Fitting

from Data_handling.vae import Encoder,Decoder, VAE , loss_function, model_train

os.chdir("Analysis")
os.getcwd()


os.chdir("../Data")  #change to data set folder and get the file names
mypath=os.getcwd()
spectra = [f for f in listdir(mypath) if (isfile(join(mypath, f))and('.fits'in f))]
spectrum_list=[]
for i in range(0,len(spectra)):
    print(str(i) + ' out of ' + str(len(spectra)))
    spectrum_list.append(Spectrum(spectra[i],parameters=True)) #at the moment there are all hr10

os.chdir('../Data')
with open('class_remote','wb') as f:
    pickle.dump(spectrum_list,f)

with open('class_remote', 'rb') as f:
    spectrum_list = pickle.load(f)
os.chdir('../Analysis')
