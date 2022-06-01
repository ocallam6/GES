#non volume preserving normalising flow
#we use gaussian mixture model for this however.

from sklearn.covariance import log_likelihood
from torch import distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from gmm_torch import GaussianMixture
from gmmflow import TorchGaussMixture

# I will need to do batch normalisation
# I will need to do batch normalisation


class FlowGMM(nn.Module):
    def __init__(self,layers,n_features,mixture_components,hidden_dims,d,means):
        super().__init__()
        
        self.layers=layers
        self.b1=torch.tensor([i for i in range(1,n_features+1)],requires_grad=False).le(d)
        self.b2=torch.tensor([i for i in range(1,n_features+1)],requires_grad=False).ge(n_features-d+1)
        self.D=n_features
        self.d=d
        #self.prior=distributions.MultivariateNormal(torch.zeros(n_features), torch.eye(n_features))
        self.prior=TorchGaussMixture(means=means)
        #self.prior=GaussianMixture(n_components=mixture_components,n_features=self.D,init_params='kmeans',covariance_type='diag')

        self.s_net=nn.ModuleList([nn.Sequential(
            nn.Linear(self.d,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],(self.D-self.d)) #d is the dimension of 1:d vector, 
                            ) for i in range(layers)])
        self.t_net=nn.ModuleList([nn.Sequential(
            nn.Linear(self.d,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],(self.D-self.d)) #d is the dimension of 1:d vector, 
                            ) for i in range(layers)])
    
    
    def forward(self,x):
        det_s=torch.zeros(len(x))
        labels=x[:,x.shape[-1]-1]
        x=x[:,:x.shape[-1]-1]
        


        loss=0.0
        det_s=torch.zeros(len(x))

        for i in range(self.layers):
            #paper recommends doing batch normalisation at each layer
            n=nn.BatchNorm1d(self.D)
            x=n(x)
            #####################################################
            #Switch the mask at each layer.
            if(i%2==0):
                b=self.b1
            else:
                b=self.b2
            
            
            x1d=x[:,b.nonzero()[:,0]]
            xdD=x[:,(~b).nonzero()[:,0]]
    
            #####################################################
            # Coupling layer

            y1d=x1d

            s,t=self.s_net[i](x1d),self.t_net[i](x1d)

            ydD=xdD*torch.exp(s)+t
            
            #######
            # Determinant of the transformation is sum of s terms

            det_s+=s.sum(-1) #each row is component of s, thats summed then summed over each value due to independence?
            x=torch.cat([y1d,ydD],-1) #loop through to the next layer
        
        y=x
        #########################################
        #loss log likelihood part
        ###
        #Fit GMM to image of f(x)
        
        gmm=self.prior
        #gmm.fit(y)
        
        log_likelihood=gmm.log_prob(y,labels)
        #log_likelihood=gmm.log_prob(y,labels).mean()#gmm.score_samples(y).sum()#gmm.log_prob(y).mean()#
        ## Loss is negative log likelihood
        loss=-1*(det_s+log_likelihood).mean()
        
        return y, gmm, loss,log_likelihood,det_s, log_likelihood.mean(),det_s.mean()



