from re import M
import sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class Pca_Fitting:
    def __init__(self,spectrum_list_learning,n_components):

        self.pca=sklearn.decomposition.PCA(n_components=n_components)
        self.learning_spectra=self.normalised_pca(spectrum_list_learning)#normalised
        self.spec_fitted=self.pca.fit_transform(self.learning_spectra) 
        self.reconstructed_spectrum=self.pca.mean_+np.matmul(self.spec_fitted,self.pca.components_)
        
    def normalised_pca(self,list_spec): 
        X_train_scaled = normalize(list_spec)
        return X_train_scaled



    def plot_orig_and_recon_w_offset(self,index):
        plt.plot(self.learning_spectra[index],label='orig norm')
        plt.plot(self.reconstructed_spectrum[index]+0.03,label='recon')
        plt.title('Comparison recon and orig')
        plt.legend()
        
    def plot_normalised(self,index):   
        plt.plot(self.learning_spectra[index],label=str(index))
        plt.title('Original Scaled')
        plt.legend() 

    def mse_fit(self):
        return np.square(self.learning_spectra-self.fitted_pca).mean()
    
    def mse_comparison_plot(self,components):
        fitted_pca=list()

        
        pca=sklearn.decomposition.PCA(n_components=components)

        spec_fitted=pca.fit_transform(self.learning_spectra)

        mse_tensor=[]
        for i in range(0,np.shape(fitted_pca)[0]):
            pca_product=np.matmul(spec_fitted[:,0:i+1],pca.components_[0:i+1])            
            fitted_pca=pca.mean_+pca_product
            mse_tensor.append((np.square(self.learning_spectra-fitted_pca).mean(axis=1)))
        mse_tensor=np.array(mse_tensor).transpose()
        mse_average=mse_tensor.mean(axis=0)


        fig, axs = plt.subplots(2,2)
        print(axs)
        
        fig.set_figheight(25)
        fig.set_figwidth(25)
        #axs[0,0].axhline(10,color='r')
        axs[0,0].set_title('Mean MSE for PCA Fit per PCA components')

        sn.lineplot(x=range(1,np.shape(fitted_pca)[0]+1),y=mse_average,ax=axs[0,0])


        axs[0,1].plot(self.learning_spectra[100],label='orig')
        axs[0,1].plot((pca.mean_+np.matmul(spec_fitted[:,0:0+1],pca.components_[0:0+1]))[100]+0.03,label='1')
        axs[0,1].plot((pca.mean_+np.matmul(spec_fitted[:,0:10+1],pca.components_[0:10+1]))[100]+0.06,label='11')
        axs[0,1].plot((pca.mean_+np.matmul(spec_fitted[:,0:20+1],pca.components_[0:20+1]))+0.09, label='21')
        axs[0,1].plot((pca.mean_+np.matmul(spec_fitted[:,0:28+1],pca.components_[0:30+1]))[100]+0.12,label='29')
        axs[0,1].legend()
        axs[0,1].set_title('Spectrum reconstructed from different number of PCA components')
        axs[1,0].plot(pca.explained_variance_ratio_)
        axs[1,0].set_title('Explained variance per component')
        axs[1,0].axvline(2,color='r',label='2')
        axs[1,0].legend()
            
        
        
