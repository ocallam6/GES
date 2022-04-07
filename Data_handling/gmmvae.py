import os
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
The encoder is the density function q_φ(z|x) which is trying to estimate the 
p_θ(z|x)
We are using factorised Gaussian Posteriors model, where we have
that q_φ(z|x)=N(z|mu,diag(sigma^2))

repar we get
epsilon ~ N(o,I)
(mu,log sigma)=NN_φ(x)
z=mu+sigma x epsilon
'''


class Encoder(nn.Module): #guide function q(z given x)
    def __init__(self, input_dim,mixture_components, z_dim, hidden_dims,dropout):
        super(Encoder,self).__init__()
        
        self.hidden1 = nn.Linear(mixture_components+input_dim, hidden_dims[0])
        self.hidden2=nn.Linear(hidden_dims[0],hidden_dims[1]) #changed from linear
        self.nn_mu = nn.Linear(hidden_dims[1], z_dim)
        self.nn_log_sigma = nn.Linear(hidden_dims[1], z_dim)


        self.dropout = nn.Dropout(p=dropout)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.softplus=nn.Softplus()
        self.softmax=nn.Softmax()


    def reparameterization(self,mean,sigma):
        epsilon=torch.randn_like(sigma) #is this the right way of doing it
        z=mean+sigma*epsilon 
        return z 

    def forward(self,x,y):
        x=torch.cat((x,y),dim=1)
        hidden_layer1=self.dropout(self.relu(self.hidden1(x)))
        hidden_layer2=self.dropout(self.relu(self.hidden2(hidden_layer1)))

        z_mu = self.nn_mu(hidden_layer2)
        z_log_sigma = self.softplus(self.nn_log_sigma(hidden_layer2))

        z=self.reparameterization(z_mu,torch.exp(0.5*z_log_sigma))
        return z, z_mu, z_log_sigma

'''The prior is the learned mixture of gaussians where mean and variance given from network depending on y'''
'''The dimension of the y vector is the mixture components dimension'''
''' The output dimension of the prior is the latent variable'''





class encoder_y(nn.Module): #guide function q(y given x)
    def __init__(self, input_dim, mixture_components, hidden_dims,dropout):
        super(encoder_y,self).__init__()
        
        self.hidden1 = nn.Linear(input_dim, hidden_dims[0])
        self.hidden2=nn.Linear(hidden_dims[0],hidden_dims[1]) #changed from linear

        self.h_gmm=nn.Linear(hidden_dims[1],mixture_components)

        self.dropout = nn.Dropout(p=dropout)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.softplus=nn.Softplus()
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        hidden_layer1=self.dropout(self.relu(self.hidden1(x)))
        hidden_layer2=self.dropout(self.relu(self.hidden2))

        qy_logits=self.h_gmm(hidden_layer2)
        qy=self.softmax(qy_logits)

        return qy_logits, qy

        

'''
The decoder is the disctribution p_θ(x|z) which uses the value of the latent variables generated from the encoders and we want to see how
different this will be.

'''



class Decoder(nn.Module): #likelihood function p(x given z)
    def __init__(self,z_dim, mixture_components,hidden_dims, output_dim,dropout):
        super(Decoder,self).__init__()
        self.hidden_y_1 = nn.Linear(mixture_components, z_dim)
        self.hidden_y_2=nn.Linear(mixture_components, z_dim) #changed from linear
        
        

        self.hidden_z_1=nn.Linear(z_dim,hidden_dims[0])
        self.hidden_z_2=nn.Linear(hidden_dims[0],hidden_dims[1])
        self.nn_out = nn.Linear(hidden_dims[1], output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.softplus=nn.Softplus()
        self.softmax=nn.Softmax(dim=1)

    def forward(self,z,y):


        #### this is p(x|z,y)
        layer1=self.relu((self.hidden_z_1(z)))
        layer2=self.dropout(self.relu(self.hidden_z_2(layer1))) # z --> nn fully connected --> softplus activation --> hidden
        x_recon_logit=self.softmax(self.nn_out(layer2))  #hidden --> nn fully connected --> sigmoid --> reconstructed spectrum
        

        #### THIS IS THE GMM PRIOR!!!! P(Z|Y)
        zm=self.self.relu(self.hidden_y_1(y))
        zv=self.self.softplus(self.hidden_y_2(y))

        return zm,zv,x_recon_logit

#Define the VAE now.

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder,mixing_components,encoder_y,prior_gmm):
        super(VAE,self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
        self.mixing_components=mixing_components
        self.encoder_y=encoder_y
        self.prior_gmm=prior_gmm
        self.sm=torch.nn.LogSoftmax(dim=1)

        self.cel=torch.nn.CrossEntropyLoss()
        self.bce = nn.BCELoss(reduction='none')
        
    def entropy(self,logits, targets):
        
        log_q = self.sm(logits)

        return -torch.sum(targets*log_q, axis=1)

    def forward(self,x):
        '''' y encoder'''
        #q(y|x)
        y_logits,y=self.encoder_y(x)
        
        
        '''Prior on the GMM'''
        p_z_given_y=self.prior_gmm(y)

        '''Encoder on gmm, takes the sampled y, x and gives a sample of z'''

        #q(z|x,y)
        z,zm,zv=self.Encoder(x,y)

    
        

        '''Generative distribution, decoder'''
        #p(x|z,y)
        zm_prior,zv_prior,x_recon_logit=Decoder(z,y)

        ent=-self.cel(y_logits,y)


        nll = -torch.mean(p_x_given_z.log_prob(x))
        kl_div_z = torch.mean(q_z.log_prob(z) - p_z_given_y.log_prob(z))
        nent = -torch.mean(self.entropy(
            qy.logits, self.sm(qy.logits)))

        loss = nll + kl_div_z + nent
        return  loss, x_r , z

'''

When doing the loss do we need our batch do be big enough so that taking the mean approximates the expectation

'''


BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = torch.sum(0.5*(x-x_hat)**2)  #nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    kldweight=0.000001
    return 10000*reproduction_loss + KLD*kldweight

def model_train(vae_spec,batch_size,optimizer,model,loss_function,epochs):
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(vae_spec):

            x = x.view(batch_size, len(x[0]))
            x = x.to(DEVICE)

            optimizer.zero_grad()

            loss, x_rec , z = model(x)
            
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Overall Loss: ", overall_loss)
    print("Finish!!")


