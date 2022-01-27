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


class Spectrum:
    #---- Constructor
    def __init__(self,spectrum_file_name,par_file,parameters=True):
        self.object, self.data=self.get_data_and_name(file=spectrum_file_name)
        if(parameters==True):
            self.parameters=self.get_parameters(name=self.object,par=par_file)
        else:
            self.parameters=np.NaN



    def get_data_and_name(self,file):
        if('Data' not in os.getcwd()):
            try:
                os.chdir("../Data")
            except:
                os.chdir("../../Data")
        spectrum=fits.open(file)
        name=spectrum[1].header['OBJECT']
        wlength=spectrum[1].data[0][0]
        flux=spectrum[1].data[0][1]
        flux_err=spectrum[1].data[0][2]

        data=pd.DataFrame(data= np.array([wlength,flux,flux_err]).transpose(),
            columns=['wavelength','flux','flux_err'])

        return name, data

    def get_parameters(self,name,par): #could maybe do a try except here in case in GES)
        try:
            for i in range (0,len(par)):
                if str(self.object)==par[i][0]:
                    return par[i]
                
            print('returning nan for parameters as no match of names')
            return np.NaN



        except:
            print("returning NAN for parameter data")
            return np.NAN

    def get_object(self):
        return self.object
    def get_columns(self):
        return self.data.columns




def learning_data(spectrum_list,parameters):
    columns=list(spectrum_list[0].data['wavelength'].astype(str))
    
    learning_spectra=[]
    parameter_lists=[]
    for name in parameters:
        parameter_array=[]    
        for i in range(0,len(spectrum_list)):
            if(name==parameters[0]):
                learning_spectra.append(spectrum_list[i].data['flux'])
            
            try:    
                parameter_array.append(spectrum_list[i].parameters[name].values[0])
            except:
                parameter_array.append(np.NaN)  
        parameter_lists.append(parameter_array)


    flux=pd.DataFrame(data=np.array(learning_spectra),columns=columns)
    teff=pd.DataFrame(data=np.array(parameter_lists).transpose(),columns=parameters)
    return pd.concat([flux,teff],axis=1)