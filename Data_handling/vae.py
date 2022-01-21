import os

import numpy as np
import torch

import torch.nn as nn

from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module): #guide function q(z given x)
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.softplus=nn.Softplus()
         
    def forward(self,x):
        #spectrum -->splits--> two fc -->
        # i might have to reshape so that the input is right dmesnion
        #x = x.reshape(-1, self.lambdas)
        
        hidden=self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))  
        return z_loc, z_scale





class Decoder(nn.Module): #likelihood function p(x given z)
    def __init__(self,z_dim, hidden_dim, output_dim):
        super(Decoder,self).__init__()

        #fully connected layers
        self.fc1=nn.Linear(z_dim,hidden_dim)
        self.fc21=nn.Linear(hidden_dim,output_dim)   #the length of an individual spectrum

        #activation functions
        self.softplus=nn.Softplus()
        self.sigmoid=nn.Sigmoid()

    def forward(self,z):
        hidden=self.softplus(self.fc1(z)) # z --> nn fully connected --> softplus activation --> hidden
        x_recon=self.sigmoid(self.fc21(hidden))  #hidden --> nn fully connected --> sigmoid --> reconstructed spectrum
        return x_recon

'''For continuous latent variables and a differentiable encoder and genera- tive model,
 the ELBO can be straightforwardly differentiated w.r.t. 
both φ and θ through a change of variables, also called the reparameterization trick (Kingma and Welling, 2014 and Rezende et al., 2014).'''

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE,self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def reparameterization(self,mean,var):
        epsilon=torch.randn_like(var)
        z=mean+var*epsilon 
        return z 

    def forward(self,x):
        mean,log_var=self.Encoder(x)
        z=self.reparameterization(mean, torch.exp(0.5*log_var))
        x_recon=self.Decoder(z)
        return x_recon, mean, log_var  









BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def model_train(vae_spec,batch_size,optimizer,model,loss_function,epochs):
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(vae_spec):

            x = x.view(batch_size, len(x[0]))
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finish!!")