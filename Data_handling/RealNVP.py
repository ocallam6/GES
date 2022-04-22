#non volume preserving normalising flow
#we use gaussian mixture model for this however.

from sklearn.covariance import log_likelihood
from torch import distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gmm_torch import GaussianMixture

# I will need to do batch normalisation

class FlowGMM(nn.Module):
    def __init__(self,layers,n_features,mixture_components,hidden_dims,d):
        super().__init__()
        self.layers=layers
        
        self.n_features=n_features
        self.d=d
        self.prior=GaussianMixture(n_components=mixture_components,n_features=self.n_features)

        self.st_net=nn.Sequential(
            nn.Linear(self.d,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],2*(self.n_features-self.d)) #d is the dimension of 1:d vector, 
                            )

    def forward(self,x):
        s_loss=torch.empty_like(x[:,self.d:])
        for i in range(self.layers):
            if(i%2==0):
                x1,x2=x[:,:self.d],x[:,self.d:]
            else:
                x1,x2=x[:,-self.d:],x[:,:-self.d] #is the flip
            
            st=self.st_net(x1)
            s,t=st[:,(self.n_features-self.d):],st[:,:(self.n_features-self.d)]

            y1=x1
            y2=x2*torch.exp(s)+t

            s_loss=torch.cat([s_loss,s.sum(dim=-1)],dim=-1)

            x=torch.cat([y1,y2],-1)
        y=x
        s_loss=s_loss.sum(dim=-1)

        gmm=self.prior.fit(y)
        log_likelihood=gmm.score_samples(y)

        loss=-1*(s_loss+log_likelihood)

        return y, gmm, loss 



n_samples = 2000

# Define distribution. 
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
X, y = noisy_moons
X = StandardScaler().fit_transform(X)

model = FlowGMM(layers=2,n_features=X.shape[-1],mixture_components=2,hidden_dims=[10,10],d=1)

# Training hyperparameters.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Increase or decrease this if you wish.
iters = 10000



train_enum = range(iters - 1)

# Initialise the minimum loss at infinity.
min_loss = float('inf')

# Iterate over the number of iterations.
for i in train_enum:
    # Sample from our "dataset". We artificially have infinitely many data points here.
    noisy_moons = datasets.make_moons(n_samples=128, noise=.05)[0].astype(np.float32)
    X = StandardScaler().fit_transform(X)
    
    optimizer.zero_grad()
    
    batch = torch.FloatTensor(noisy_moons)
    y,gmm,loss = model(batch)
    # If the loss is lower than anything already encountered, consider that the "best model".
    if loss.item() < min_loss:
        bestmodel = model
    
    # Backpropagation.
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print('Iter {}, loss is {:.3f}'.format(i, loss.item()))