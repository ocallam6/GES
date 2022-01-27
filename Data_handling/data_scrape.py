from astropy.io import fits
import os


import numpy as np


import pickle
import time 


from Data_handling.spectra_object import Spectrum, learning_data
from multiprocessing import Pool 



def par_to_numpy_and_save(save_location):

    if('Parameter_files' not in os.getcwd()):
        try:
            os.chdir("../Parameter_files")
        except:
            os.chdir("Parameter_files")
    print('opening fits file')
    par=fits.open("GES_iDR6_WG15_Recommended_with_sflags__mode_normal_091221.fits")
    print(os.getcwd())

    formatting=list()
    for i in range(0,len(par[1].data)):
        print(i)
        formatting.append(np.array(par[1].data[i]))
        

    np.save(save_location,formatting,allow_pickle=True)





def create_spectrum_list(spectra,parameter_array):
    print('this requires that par_to_numpy_and_save has been run and requires \n that in the spectrum class load the numpy save location from notebook')
    start=time.time()
    spectrum_list=[]
    spectrum_list=[Spectrum(spectrum,parameters=False,par_file=parameter_array) for spectrum in spectra]
    print(time.time()-start)

    with open('class_f','wb') as f:
        pickle.dump(spectrum_list,f)

#par_to_numpy_and_save('par_array.npy')
