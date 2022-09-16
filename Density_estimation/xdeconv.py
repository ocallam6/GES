"""
Extreme deconvolution solver

This follows Bovy et al.
http://arxiv.org/pdf/0905.2979v2.pdf

Arbitrary mixing matrices R are not yet implemented: currently, this only
works with R = I.
"""
from queue import Empty
from time import time
from matplotlib.pyplot import axes

import numpy as np
from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as logsumexp
except ImportError:
    from scipy.misc import logsumexp as logsumexp

from scipy.stats import multivariate_normal

from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state

from astroML.utils import log_multivariate_gaussian

class XDGMM(BaseEstimator):
    """Extreme Deconvolution

    Fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    n_components: integer
        number of gaussian components to fit to the data
    max_iter: integer (optional)
        number of EM iterations to perform (default=100)
    tol: float (optional)
        stopping criterion for EM iterations (default=1E-5)

    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
    def __init__(self, n_components, max_iter=100, tol=1E-5, verbose=False,
                    random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        # model parameters: these are set by the fit() method
        self.V = None
        self.mu = None
        self.alpha = None


    def fit(self, X, Xerr, R):
        """Fit the XD model to data

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)
        R : array_like
            (TODO: not implemented)
            Transformation matrix from underlying to observed data.  If
            unspecified, then it is assumed to be the identity matrix.
        """

        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        # initialize components via a few steps of GaussianMixture              #do an initial guess on the full data
        # this doesn't take into account errors, but is a fast first-guess
        
        gmm = GaussianMixture(self.n_components, max_iter=100,
                                covariance_type='full',
                                random_state=self.random_state).fit(X)
        
        self.mu = gmm.means_
        self.alpha = gmm.weights_
        self.V = gmm.covariances_
        
        logL = self.logL(X, Xerr,R)

        for i in range(self.max_iter):
            t0 = time()
            self._EMstep(X, Xerr,R)
            logL_next = self.logL(X, Xerr,R)
            t1 = time()

            if self.verbose:
                print("%i: log(L) = %.5g" % (i + 1, logL_next))
                print("    (%.2g sec)" % (t1 - t0))

            if logL_next < logL + self.tol:
                break
            logL = logL_next

        return self

    def logprob_a(self, X, Xerr,R):
        """
        Evaluate the probability for a set of points

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)

        Returns
        -------
        p: ndarray
            Probabilities.  shape = (n_samples,)
        """
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape
        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]



        Xerr = Xerr[:, np.newaxis, :, :]
        R=R[:,np.newaxis,:,:]

 
        #so the log_multivariate gaussian requires that the mean has just n_component vectors in it
        RVRt=np.matmul(np.matmul(R[:],self.V[:]),R.transpose(0,1,3,2)[:])   #we dont do transpose as all of our matrices are diagonal
        

        Rmu=np.matmul(R,self.mu[np.newaxis,:,:,np.newaxis])
        #Rmu=R[:]*self.mu[np.newaxis,:,:,np.newaxis]


        T = RVRt+ Xerr
        



        return log_multivariate_gaussian(X, Rmu.reshape(Rmu.shape[0:3]), T) + np.log(self.alpha)

    def logL(self, X, Xerr,R):
        """Compute the log-likelihood of data given the model

        Parameters
        ----------
        X: array_like
            data, shape = (n_samples, n_features)
        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)

        Returns
        -------
        logL : float
            log-likelihood
        """
        # we need to run through all the different R values
        return np.sum(logsumexp(self.logprob_a(X, Xerr,R), -1))

    def _EMstep(self, X, Xerr,R):
        """
        Perform the E-step (eq 16 of Bovy et al)
        """


        n_samples, n_features = X.shape


        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        R=R[:,np.newaxis,:,:]

        RVRt=np.matmul(np.matmul(R[:],self.V[:]),R.transpose(0,1,3,2)[:])   #we dont do transpose as all of our matrices are diagonal
        
        Rmu=np.matmul(R,self.mu[np.newaxis,:,:,np.newaxis])

        T = RVRt+ Xerr


        w_m = X - Rmu.reshape(Rmu.shape[0:3])


        # ------------------------------------------------------------
        #  compute inverse of each covariance matrix T
        
        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                        n_features, n_features])
        Tinv = np.array([linalg.inv(T[i])
                            for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)

        # ------------------------------------------------------------
        #  evaluate each mixture at each point
        N = np.exp(log_multivariate_gaussian(X, Rmu.reshape(Rmu.shape[0:3]), T, Vinv=Tinv))

        # ------------------------------------------------------------
        #  E-step:
        #   compute q_ij, b_ij, and B_ij
        q = (N * self.alpha) / np.dot(N, self.alpha)[:, None]



        tmp = np.sum(np.matmul(Tinv , w_m[:, :, :,np.newaxis]), -1)  #sum just gets rid of that dummy axis, is essentially a reshape function
        
        
        
        
        b = self.mu + np.sum(np.matmul(np.matmul(self.V,R.transpose(0,1,3,2)) , tmp[:, :, :,np.newaxis]), -1)  #might have a newaxis problem here3




        tmp = np.matmul(np.matmul(np.matmul(R.transpose(0,1,3,2),Tinv[:, :, :, :]),R)
                        , self.V[:, :, :])     #again could have a newaxis problem here



                        
        B = self.V - np.matmul(self.V[:, :, :]
                            , tmp[:, :, :])


        # ------------------------------------------------------------
        #  M-step:
        #   compute alpha, m, V
        qj = q.sum(0)

        self.alpha = qj / n_samples

        self.mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]

        m_b = self.mu - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        self.V = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]


    def sample(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.random_state
        rng = check_random_state(random_state)  # noqa: F841
        shape = tuple(np.atleast_1d(size)) + (self.mu.shape[1],)
        npts = np.prod(size)  # noqa: F841

        alpha_cs = np.cumsum(self.alpha)
        r = np.atleast_1d(np.random.random(size))
        r.sort()

        ind = r.searchsorted(alpha_cs)
        ind = np.concatenate(([0], ind))
        if ind[-1] != size:
            ind[-1] = size

        draw = np.vstack([np.random.multivariate_normal(self.mu[i],
                                                        self.V[i],
                                                        (ind[i + 1] - ind[i],))
                            for i in range(len(self.alpha))])

        return draw.reshape(shape)
    
    def prob_z_given_w(self, X,Xerr,R):
        
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape
        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]



        Xerr = Xerr[:, np.newaxis, :, :]
        R=R[:,np.newaxis,:,:]
        
        RVRt=np.matmul(np.matmul(R[:],self.V[:]),R.transpose(0,1,3,2)[:])   #we dont do transpose as all of our matrices are diagonal
        
        Rmu=np.matmul(R,self.mu[np.newaxis,:,:,np.newaxis])
        Rmu=Rmu.reshape(Rmu.shape[0:3])

        T = RVRt+ Xerr

        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                        n_features, n_features])
        Tinv = np.array([linalg.inv(T[i])
                            for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)


        output=[]
        components=np.array([[0.0]*self.n_components]*len(X)).transpose()

        for j in range(0,len(X)):
            sum=0
            for i in range(0,self.n_components):
                components[i,j]=self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
                sum=sum+self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
            components[:,j]=components[:,j]/sum
        return components.transpose()

    def final_prob(self, X,Xerr,R,prob_true,prob_false,true_index):
        
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape
        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]



        Xerr = Xerr[:, np.newaxis, :, :]
        R=R[:,np.newaxis,:,:]
        
        RVRt=np.matmul(np.matmul(R[:],self.V[:]),R.transpose(0,1,3,2)[:])   #we dont do transpose as all of our matrices are diagonal
        
        Rmu=np.matmul(R,self.mu[np.newaxis,:,:,np.newaxis])
        Rmu=Rmu.reshape(Rmu.shape[0:3])

        T = RVRt+ Xerr

        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                        n_features, n_features])
        Tinv = np.array([linalg.inv(T[i])
                            for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)


        output=[]
        components=np.array([[0.0]*self.n_components]*len(X)).transpose()

        for j in range(0,len(X)):
            sum=0
            for i in range(0,self.n_components):
                if i == true_index:
                    components[i,j]=prob_true[j]*self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
                    sum=sum+prob_true[j]*self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
                if i != true_index:
                    components[i,j]=prob_false[j]*self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
                    sum=sum+prob_false[j]*self.alpha[i]*multivariate_normal.pdf(X[j],Rmu[j,i,:],T[j,i,:,:])
            components[:,j]=components[:,j]/sum
        return components.transpose()
