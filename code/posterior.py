import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
import scipy.special

import astropy.io.fits as fits
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from clarsach import ARF, RMF


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly-spaced Gaussian Features for 1D input
    from: https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html

    """
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)



class PoissonPosterior(object):
    
    def __init__(self, x, y, n_gauss, model=None, width_factor=0.4,
                 arf=None, rmf=None, xmax=None, apply_response=True):
        """
        Poisson Posterior for a single 1D X-ray spectrum.
        The spectrum can be modelled as a combination of a 
        set of Gaussian basis functions plus another parametric 
        model. It is also possible to *only* use a parametric model
        (when `n_gauss == 0`) or *only* use the a combination of 
        Gaussians (if `model == None`). 
        
        `arf` and `rmf` are expected to be instances of the respective
        Clarsach classes. If they are set to None, no responses will be 
        applied to the model before calculating the log-likelihood.
        
        Parameters
        -----------
        x, y: numpy.ndarrays
            The energies and counts of the observed spectrum
        
        n_gauss : int
            The number of polynomial terms (excluding the intercept)
        
        width_factor : float, optional, default 0.4
            The width factor for the Gaussan basis functions.
            Should probably be a fit parameter?
        
        model : `astropy.modeling.models` instance or `None`
            An astropy model. If set to `None`, only the 
            polynomial model will be calculated
            
        include_bias : bool, default: True
            boolean flag to decide whether to set an intercept
            weight (the x^0 term)
        
        arf, rmf: clarsach.ARF and clarsach.RMF instances
            The responses for the spectrum in `x` and `y`, 
            used to convert the source spectrum into the counts 
            observed by the detector.
            
        xmax: float
            The maximum energy to consider. Useful when only part 
            of the spectrum is of interest.
        
        Attributes
        ----------
        p : scipy.PolynomialFeatures instance
            sets the polynomial terms
            
        pft : (n_gauss, N) numpy.ndarray
        
        """
        
        if n_gauss == 0 and model is None:
            raise AttributeError("Either n_gauss must be non-zero or another model must be set!")
        
        self.x = x
        self.y = y
        
        self.arf = arf
        self.rmf = rmf
        
        self.n_gauss = n_gauss
        self.model = model
        self.xmax = xmax
        self.width_factor = width_factor
        self.apply_response = apply_response
        
        if self.xmax is not None:
            self.max_ind = self.x.searchsorted(self.xmax)
        else:
            self.max_ind = np.max(self.x.shape)+1
        
        if self.n_gauss > 0:
            # compute polynomial terms
            self._compute_gauss()
            
        if self.model is not None:
            self.npar = 0
            for pname in self.model.param_names:
                if not self.model.fixed[pname]:
                    self.npar += 1

        
    def _compute_gauss(self):
        self.p = GaussianFeatures(N=self.n_gauss, width_factor=self.width_factor)

        self.pft = self.p.fit_transform(self.x[:, np.newaxis])
        
        return
    
    def _apply_response(self, mean_model):
        """
        If any responses are given, apply them 
        to the model.
        """
        if self.arf is None and self.rmf is None:
            return mean_model
        
        else:
            model_arf = self.arf.apply_arf(mean_model)
            model_rmf = self.rmf.apply_rmf(model_arf)
            
            return model_rmf
        
    def _compute_mean_model(self, pars):
        # if no polynomial is used, initialize the mean 
        # model as a row of zeros
        if self.model is not None:
            _fitter_to_model_params(self.model, pars[-self.npar:])
            mean_model = self.model(self.x)
        
        else:
            mean_model = np.ones_like(self.x)
        
        if self.n_gauss > 0:
            # else get the weights vector out of the 
            # parameter vector and compute the polynomial
            
            # get the weights out of the parameter vector
            w = pars[:self.n_gauss]
            
            # compute polynomial mean model 
            mean_model *= np.dot(self.pft, w)
                        
        # if responses are given, apply them before 
        # calculating the likelihood
        if self.apply_response:
            model_counts = self._apply_response(mean_model)
        else:
            model_counts = mean_model

        return model_counts
    
    def logprior(self, pars):
        if (np.any(pars) < -10000) or (np.any(pars) > 10000):
            return -np.inf
        
        else:
            return 0.0

    def loglikelihood(self, pars, neg=False):
        """
        Evaluate the Poisson likelihood for a set of parameters.
        
        Parameters
        ----------
        pars : iterable
            A set of parameters to evaluate. The first `n_poly`
            parameters are the weights for the Polynomials, 
            the rest correspond to the parameters in `model`
        
        neg : bool, default False
            If True, return the negative log-likelihood.
            Set this to True for doing optimization
            (where you'll need the negative log-likelihood)
        
        Returns
        -------
        llike : float
            The log-likelihood of the data given the model.
        
        """
        
        model_counts = self._compute_mean_model(pars)
        
        model_counts[model_counts == 0.0] = 1e-20 
        
        llike = -np.nansum(model_counts[:self.max_ind]) + \
                np.nansum(self.y[:self.max_ind]*np.log(model_counts[:self.max_ind])) - \
                np.nansum(scipy.special.gammaln(self.y[:self.max_ind] + 1))
    
        if not np.isfinite(llike):
            llike = -np.inf

        if neg:
            return -llike
        else:
            return llike
    
    def logposterior(self, pars, neg=False):
        
        lp = self.logprior(pars) + self.loglikelihood(pars, neg=False)
        
        if not np.isfinite(lp):
            lp = -np.inf

        if neg:
            return -lp
        else:
            return lp
        
    
    def __call__(self, pars, neg=False):
        return self.logposterior(pars, neg=neg)


class TwoSpectrumPoissonPosterior(object):
    
    def __init__(self, n_gauss, x, ybkg, ysrc=None, bkg_model=None, src_model=None, 
                 width_factor=0.4, arf=None, rmf=None, 
                 xmax=None, apply_response=True):
        """
        Poisson Posterior for a single 1D X-ray spectrum.
        The spectrum can be modelled as a combination of a 
        set of Gaussian basis functions plus another parametric 
        model. It is also possible to *only* use a parametric model
        (when `n_gauss == 0`) or *only* use the a combination of 
        Gaussians (if `model == None`). 
        
        `arf` and `rmf` are expected to be instances of the respective
        Clarsach classes. If they are set to None, no responses will be 
        applied to the model before calculating the log-likelihood.
        
        Parameters
        -----------
        x : numpy.ndarray
            The energies of the observed spectrum

        n_gauss : int
            The number of polynomial terms (excluding the intercept)

        ybkg: numpy.ndarray
            The background spectrum

        ysrc: numpy.ndarray, optional
            The source spectrum
        
        width_factor : float, optional, default 0.4
            The width factor for the Gaussan basis functions.
            Should probably be a fit parameter?
        
        model : `astropy.modeling.models` instance or `None`
            An astropy model. If set to `None`, only the 
            polynomial model will be calculated
            
        width_factor : float
            The width factor for the Gaussian basis functions.
            Could possibly become a fitting parameter 
            in a future version?
        
        arf, rmf: clarsach.ARF and clarsach.RMF instances
            The responses for the spectrum in `x` and `y`, 
            used to convert the source spectrum into the counts 
            observed by the detector.
            
        xmax: float
            The maximum energy to consider. Useful when only part 
            of the spectrum is of interest.
            
        apply_arf : bool, optional, default True
            If False, don't apply the responses to the model spectrum.
            Useful for debuggung.
        
        Attributes
        ----------
        p : scipy.PolynomialFeatures instance
            sets the polynomial terms
            
        pft : (n_gauss, N) numpy.ndarray
        
        """
        
        if n_gauss == 0 and model is None:
            raise AttributeError("Either n_gauss must be non-zero or another model must be set!")
        
        self.x = x
        self.ybkg = ybkg
        self.ysrc = ysrc
        
        self.arf = arf
        self.rmf = rmf
        
        self.n_gauss = n_gauss
        self.bkg_model = bkg_model
        self.src_model = src_model
        
        self.xmax = xmax
        self.width_factor = width_factor
        self.apply_response = apply_response
        
        if self.xmax is not None:
            self.max_ind = self.x.searchsorted(self.xmax)
        else:
            self.max_ind = np.max(self.x.shape)+1
        
        if self.n_gauss > 0:
            # compute polynomial terms
            self._compute_gauss()

        self.bkg_npar = 0
        if self.bkg_model is not None:
            for pname in self.bkg_model.param_names:
                if not self.bkg_model.fixed[pname]:
                    self.bkg_npar += 1
                    
        self.src_npar = 0
        if self.src_model is not None:
            for pname in self.src_model.param_names:
                if not self.src_model.fixed[pname]:
                    self.src_npar += 1
        
    def _compute_gauss(self):
        self.p = GaussianFeatures(N=self.n_gauss, width_factor=self.width_factor)

        self.pft = self.p.fit_transform(self.x[:, np.newaxis])
        
        return
    
    def _apply_response(self, bkg_model, source_model=None):
        """
        If any responses are given, apply them 
        to the model.
        """
        if self.arf is None and self.rmf is None:
            return bkg_model, source_model

        else:
            bkg_arf = self.arf.apply_arf(bkg_model)
            bkg_rmf = self.rmf.apply_rmf(bkg_arf)
            
            if source_model is not None:
                source_arf = self.arf.apply_arf(source_model)
                source_rmf = self.rmf.apply_rmf(source_arf)
            
                return bkg_rmf, source_rmf
            else:
                return bkg_rmf, None
                
        
    def _compute_mean_model(self, pars):
        # if no polynomial is used, initialize the mean 
        # model as a row of zeros
        if self.bkg_model is not None:
            # background model parameters are the second-to-last few in the list:
            if self.src_model is not None:
                bkg_pars = pars[-self.bkg_npar-self.src_npar:-self.src_npar]
            else:
                bkg_pars = pars[-self.bkg_npar-self.src_npar:]
            _fitter_to_model_params(self.bkg_model, bkg_pars)
            mean_model = self.bkg_model(self.x)
        
        else:
            mean_model = np.ones_like(self.x)
        
        if self.n_gauss > 0:
            # else get the weights vector out of the 
            # parameter vector and compute the polynomial
            
            # get the weights out of the parameter vector
            w = pars[:self.n_gauss]
            
            # compute polynomial mean model 
            mean_model *= np.dot(self.pft, w)
            
        if self.src_model is not None:
            # background model parameters are the second-to-last few in the list:
            src_pars = pars[-self.src_npar:]
            _fitter_to_model_params(self.src_model, src_pars)
            source_model = self.src_model(self.x) + mean_model
        
        else:
            source_model = None

        # if responses are given, apply them before 
        # calculating the likelihood
        if self.apply_response:
            bkg_counts, source_counts = self._apply_response(mean_model, source_model)
        else:
            bkg_counts = mean_model
            source_counts =  source_model
            
        return bkg_counts, source_counts
    
    def logprior(self, pars):
        if (np.any(pars) < -10000) or (np.any(pars) > 10000):
            return -np.inf
        
        else:
            return 0.0

    def loglikelihood(self, pars, neg=False):
        """
        Evaluate the Poisson likelihood for a set of parameters.
        
        Parameters
        ----------
        pars : iterable
            A set of parameters to evaluate. The first `n_poly`
            parameters are the weights for the Polynomials, 
            the rest correspond to the parameters in `model`
        
        neg : bool, default False
            If True, return the negative log-likelihood.
            Set this to True for doing optimization
            (where you'll need the negative log-likelihood)
        
        Returns
        -------
        llike : float
            The log-likelihood of the data given the model.
        
        """
        
        bkg_counts, source_counts = self._compute_mean_model(pars)

        bkg_counts[bkg_counts == 0.0] = 1e-20 

        llike = -np.nansum(bkg_counts[:self.max_ind]) + \
                np.nansum(self.ybkg[:self.max_ind]*np.log(bkg_counts[:self.max_ind])) - \
                np.nansum(scipy.special.gammaln(self.ybkg[:self.max_ind] + 1))
        
        if self.ysrc is not None:
            source_counts[source_counts == 0.0] = 1e-20 

            llike += -np.nansum(source_counts[:self.max_ind]) + \
                    np.nansum(self.ysrc[:self.max_ind]*np.log(source_counts[:self.max_ind])) - \
                    np.nansum(scipy.special.gammaln(self.ysrc[:self.max_ind] + 1))
    
    
        if not np.isfinite(llike):
            llike = -np.inf

        if neg:
            return -llike
        else:
            return llike
    
    def logposterior(self, pars, neg=False):
        
        lp = self.logprior(pars) + self.loglikelihood(pars, neg=False)
        
        if not np.isfinite(lp):
            lp = -np.inf

        if neg:
            return -lp
        else:
            return lp

    def calculate_aic(self, pars):
        """
        Calculate the Akaike Information Criterion.
        """
        loglike = self.loglikelihood(pars, neg=False)
        aic = 2.*len(pars) - 2.*loglike
        return aic
    
    def calculate_bic(self, pars):
        """
        Calculate the Bayesian Information Criterion
        """
        loglike = self.loglikelihood(pars, neg=False)
        bic = np.log(self.x.shape[0])*len(pars) - 2.*loglike
        return bic

    def __call__(self, pars, neg=False):
        return self.logposterior(pars, neg=neg)
