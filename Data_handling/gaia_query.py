
from logging import error
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier



Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3
Gaia.ROW_LIMIT=-1
Vizier.ROW_LIMIT = -1






def gaia_cone(right_as_center,dec_center,diam):    #input is the center of the cluster and its diameter

    coord = SkyCoord(right_as_center,dec_center, unit=(u.deg, u.deg))
    rad = u.Quantity(diam, u.deg)  #scanning radius, it is twice the diameter of the 
    r = Gaia.cone_search_async(coordinate=coord, radius=rad, verbose=True)   #This is the cone search radius
    
    #Gaia Query
    gaia_edr3=r.get_results()   #etting the tables from server
    gaia_edr3=gaia_edr3[np.argsort(gaia_edr3['source_id'])]
    gaia_edr3 = gaia_edr3.to_pandas()
    #Gets rid of unmeasured parallax
    print('warning, getting rid of very negaitve parallax')
    gaia_edr3=gaia_edr3[gaia_edr3['parallax']>=-1000].reset_index()
    return gaia_edr3
    
def bailer_jones_cone(right_as_center,dec_center,diam):  
    coord = SkyCoord(right_as_center,dec_center, unit=(u.deg, u.deg))
    rad = u.Quantity(diam, u.deg)  #scanning radius, it is twice the diameter of the 
    r = Gaia.cone_search_async(coordinate=coord, radius=rad, verbose=True) 
    #Next we want the bailer-jones values for the same region, unlimited row limits now
    Vizier.ROW_LIMIT = -1
    bailer = Vizier.query_region(coord,
                                 radius=rad,
                                 catalog='I/352/gedr3dis')[0]                            
    bailer=bailer[np.argsort(bailer['Source'])] #making sure match in concat
    bailer=bailer.to_pandas()
    return bailer.reset_index()

def concatenate_gaia_bailer(gaia_data,bailer_data):    

    print(len(gaia_data),' and ',len(bailer_data))
    

    ###Boolean variables
    bool1=(False in (gaia_data['source_id'].values==bailer_data['Source'].values))
    bool2=(len(gaia_data)==len(bailer_data))
    bool3=bool2 and (not bool1)
 
    #Should have more error messages and break this up more

    if(bool3): #making sure it all matches
        
        total_gaia = pd.concat([gaia_data,bailer_data], axis=1, join="inner")
        return total_gaia
    else:
        print('error in sizes')


        

# We also find the gradient of each of the data points by sorting the values and calculating gradient
def add_gradient_for_parameter(input_data, parameter):
    try:
        
        gradient_name=parameter+'_gradient'

        input_data=input_data.sort_values(by=[parameter])
        paramater_data_sorted_array=input_data[parameter].values

        input_data[gradient_name]=np.gradient(paramater_data_sorted_array)

        return input_data.sort_index()  #return back to original sorting
        #this could be computationally heavy


    except:
        print('Type error/column name error in add_gradient_for_parameter')


#next we combine the last two and tag the data frame if a value is in the required region