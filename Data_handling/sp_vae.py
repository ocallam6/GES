
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from collections import defaultdict

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        
        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(8, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
            
        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 8)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        
        return x
        
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var), (q_z, p_z), z, x_
    
    
def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.view(batch_size, x.shape[0]))
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def model_trains(vae_spec, batch_size,optimizer,model,epochs):
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x_mb in enumerate(vae_spec):

                optimizer.zero_grad()
                
                # dynamic binarization
                #x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()

                _, (q_z, p_z), _, x_mb_ = model(x_mb)

                loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb).sum(-1).mean()

                if model.distribution == 'normal':
                    loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
                elif model.distribution == 'vmf':
                    loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
                else:
                    raise NotImplemented

                loss = loss_recon + loss_KL
                overall_loss+=loss.item()
                loss.backward()
                optimizer.step()
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Overall Loss: ", overall_loss)
    print("Finish!!")
            
            
