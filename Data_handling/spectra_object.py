#Here we are going to create a class for an object given a name for the object, it will use construftors
#which create object with name, all the sprectrum data and parameters
#there will also be a function with being able to create the data from learning

#There is till plenty to do: parameter namem columns
# need to include the hr10 setting and giraffe
# Need to include units
# Need a cleaning data utility

from unicodedata import name
import pandas as pd
from astropy.io import fits
import os
from os import listdir
from os.path import isfile, join
import numpy as np

'''    THIS FILE IS VERY SPECIFIC TO THE FORMAT THAT HAS BEEN DOWNLOADED AND DOESNT HAVE MUCH USE 
ANYWEHRE ELSE  '''

def get_data(file):
    try:
        file=file[:3]+'3'+file[3:]+'s'
        spectrum=fits.open(file)
        #name=spectrum[1].header['OBJECT']
        wlength=spectrum[1].data[0][0]
        flux=spectrum[1].data[0][1]
        flux_err=spectrum[1].data[0][2]

        data=pd.DataFrame(data= np.array([wlength,flux,flux_err]).transpose(),
            columns=['wavelength','flux','flux_err'])

        return data

    except:
        return np.NAN
