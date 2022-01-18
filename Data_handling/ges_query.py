from astroquery.eso import Eso
import os
from astropy.io import ascii
import socket
eso = Eso()

#-------This function simply gets te names of objects from the Gaia ESO surevey------#

def table_query_names(giraffe_setting):
    eso.login("ocallam6")


    eso.ROW_LIMIT=200000000000000000
    print('Query begin')
    table = eso.query_surveys('GAIAESO', cache=False)
    print('Length of table queried: '+str(table))
    giraffe_table=table[table['Instrument']=='GIRAFFE']
    print('Length of giraffe table queried: '+str(giraffe_table))

    if(giraffe_setting=='HR10'):
        hr10_table=giraffe_table[giraffe_table['Wavelength']==('533.400..561.100')]
        
        names=[]
        f = open("hr10_names.txt", "a")
        for i in range(0, len(hr10_table)):
            names.append(hr10_table['ARCFILE'][i])
            f.write(str(hr10_table['ARCFILE'][i])+'\n')
        f.close()
        print('Length of final list: '+str(len(names)))
        print('Data written to ' + str(os.getcwd()))
    elif(giraffe_setting=='HR21'):
        hr21_table=giraffe_table[giraffe_table['Wavelength']==('847.500..898.200')]
        
        names=[]
        f = open("hr21_names.txt", "a")
        for i in range(0, len(hr21_table)):
            names.append(hr21_table['ARCFILE'][i])
            f.write(str(hr21_table['ARCFILE'][i])+'\n')
        f.close()
        print('Length of final list: '+str(len(names)))
        print('Data written to ' + str(os.getcwd()))
    
    else:
        print("Error in giraffee setting")

    
    
    
#-----------------------------------------------------------------------------------#



