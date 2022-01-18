import sklearn
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class Pca_Fitting:
    def __init__(self,spectrum_list_learning,n_components):

        self.pca=sklearn.decomposition.PCA(n_components=n_components)
        self.spec_fitted=self.pca.fit_transform(spectrum_list_learning)
        self.reconstructed_spectrum=self.pca.mean_+np.matmul(self.spec_fitted,self.pca.components_)
        self.learning_spectra=spectrum_list_learning
        

    def plot_orig_and_recon_w_offset(self,index):
        plt.plot(self.learning_spectra[index],label='orig')
        plt.plot(self.reconstructed_spectrum[index]+200,label='recon')
        plt.title('Comparison recon and orig')
        plt.legend()
        

    def mse_fit(self):
        return np.square(self.learning_spectra-self.fitted_pca).mean()
    
    def mse_comparison_plot(self,components):
        fitted_pca=list()
        for i in range(1,components):
            
            pca=sklearn.decomposition.PCA(n_components=i)

            spec_fitted=pca.fit_transform(self.learning_spectra)
            pca_product=np.matmul(spec_fitted,pca.components_)            
            fitted_pca.append(pca.mean_+pca_product)  
            mse_tensor=[]
        for i in range(0,np.shape(fitted_pca)[0]):
            mse_tensor.append((np.square(self.learning_spectra-fitted_pca[i]).mean(axis=1)))
        mse_tensor=np.array(mse_tensor).transpose()
        mse_average=mse_tensor.mean(axis=0)


        fig, axs = plt.subplots(2,2)
        print(axs)
        
        fig.set_figheight(25)
        fig.set_figwidth(25)
        axs[0,0].axhline(10,color='r')
        axs[0,0].set_title('Mean MSE for PCA Fit per PCA components')

        sn.lineplot(x=range(1,np.shape(fitted_pca)[0]+1),y=mse_average,ax=axs[0,0])


        axs[0,1].plot(self.learning_spectra[100],label='orig')
        axs[0,1].plot(fitted_pca[0][100]+300,label='1')
        axs[0,1].plot(fitted_pca[10][100]+600,label='9')
        axs[0,1].plot(fitted_pca[20][100]+900, label='19')
        axs[0,1].plot(fitted_pca[28][100]+1200,label='27')
        axs[0,1].legend()
        axs[0,1].set_title('Spectrum reconstructed from different PCA components')
        axs[1,0].plot(pca.explained_variance_ratio_)
        axs[1,0].set_title('Explained variance per component')
        axs[1,0].axvline(2,color='r',label='2')
        axs[1,0].legend()
            
        
        
