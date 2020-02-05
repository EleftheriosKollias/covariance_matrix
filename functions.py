#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:01:36 2020

@author: viniciussaurin
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from contextlib import contextmanager
import sys, os
import math
#from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype == "returns":
        name = f"{weighting}_rets" 
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")
    
    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew


def pca_cluster(r, n_factors=3, n_stocks_pick=10, random_state=100):
    """
    Computes the n_factors leading principal components, then clusterize them,
    in order to get the closest stock to each of cluster_center
    Rerurns the input dataframe with only the selected columns
    """
    # PCA
    sklearn_pca = sklearnPCA(n_components=n_factors)
    pca = sklearn_pca.fit(r)
    loadings = pd.DataFrame(pca.components_,columns=r.columns).T
    # Clustering
    model = KMeans(n_clusters=n_stocks_pick,random_state=random_state).fit(loadings)
    transformation = pd.DataFrame(model.transform(loadings),index=loadings.index)
    cols = list(transformation.idxmin())
    return r[cols]

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    vol = (weights.T @ covmat @ weights)**0.5
    return vol 


def msr(riskfree_rate, er, cov, bounds=(0.0,1.0)):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = (bounds,) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x



def gmv(cov,bounds=(0.0,1.0)):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov, bounds)

def weight_gmv(r, cov_estimator=sample_cov, bounds=(0.0,1.0), **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov, bounds)


def backtest_ws(r, estimation_window=60, weighting=weight_ew, stock_selector=None, verbose=False, modified=False, freq='monthly', **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
        
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    
    
    if stock_selector != None:  
        weights = pd.DataFrame(index=r.iloc[estimation_window:].index, columns=r.columns)
        for win in windows:
            r1 = pca_cluster(r.iloc[win[0]:win[1]])
            w = pd.DataFrame(weight_ew(r1.iloc[win[0]:win[1]])).T
            for i in range(len(w.columns)):
                weights.loc[:,w.columns[i]].iloc[win[0]] = w.loc[:,w.columns[i]].iloc[0]
    else:  
        if freq == 'annually':
            weights=[]
            for win in windows:
                if r.iloc[win[1]].name.month == 1:
                    weight = weighting(r.iloc[win[0]:win[1]], **kwargs)
                    weights.append(weight)
                else:
                    weights.append(weight)
            
        else:
            weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
            # convert List of weights to DataFrame
        weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    
    if not modified:
        return returns
    else:
        return [returns,weights]
    
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")



def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def enc(weights):
    """
    Returns the effective number of constituents
    """
    if weights.ndim == 1:
        return round(1/(weights**2).sum().mean(),3)
    return round(1/(weights**2).sum(axis=1).mean(),3)

def risk_contribution(w,cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib

def encb(r, weights, cov=sample_cov):
    """
    Returns the effective number of correlated bets, which is like the ENC but
    measuring the risk_contribution
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
        
    if weights.ndim == 1:
        weights.reshape(len(weights),1)
        return round(1/(risk_contribution(weights, cov(r))**2).sum(),3)
    else:
        p = np.zeros(weights.shape[0])
        for i in range(weights.shape[0]):
            weights[i,:].reshape(len(weights[1,:]),1)
            if (risk_contribution(weights[i,:], cov(r))**2).sum() > 0:
                p[i] = 1/(risk_contribution(weights[i,:], cov(r))**2).sum()
            else:
                p[i] = 0
        return round(p.mean(),3)
    
def ewma_cov(r, decay_parameter=0.97, modified=False):
    """
    Estimates a covariance matrix using the exponentially weighted average, since the volatility
    is non-stationary
    If modified== True computes the covariance matrix assuming returns mean is zero
    """
    
    n_periods = r.shape[0]
    lambda_vector = np.array([decay_parameter**(n_periods - t) for t in range(1, n_periods+1)])
    lambda_vector = lambda_vector.reshape(len(lambda_vector),1)
    alpha = lambda_vector/lambda_vector.sum()
    
    if modified:
        return pd.DataFrame(np.dot(r.T, r * alpha))
    else:
        r_centered = r - r.mean(axis=0)
        return pd.DataFrame(np.dot(r_centered.T, r_centered * alpha))

def orthogonal_ewma_cov(r, **kwargs):
    """
    Computes the exponentially weighted volatility for each of the principal
    components
    """
    
    nrow = r.shape[0]
    ncol = r.shape[1]
    r_std = StandardScaler().fit_transform(r)/np.sqrt(nrow)
    X_T_X = (r_std.T @ r_std)
    eigvals, W = np.linalg.eig(X_T_X)
    p = r_std @ W
    sigma = np.array(r.std(ddof=0)).reshape(ncol,1)
    A = W * sigma
    cov_mat_pc = ewma_cov(p, **kwargs)
    V = A @ cov_mat_pc @ A.T
    return pd.DataFrame(V)

@contextmanager
def suppress_stdout():
    """
    Hide the output
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def best_garch_model(TS, maxp, maxo, maxq, dist):
    """
    Given maximum values of p, o and q this function computes the best model in
    terms of AIC
    """
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    p_vector = range(maxp+1)
    o_vector = range(maxo+1)
    q_vector = range(maxq+1)

    for i in p_vector:
        for k in o_vector:
            for j in q_vector:
                try:
                    with suppress_stdout(): 
                        tmp_mdl = arch_model(TS, vol= 'GARCH', p=i, o=k, q=j, dist=dist).fit()
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, k, j) # p, o, q
                            best_mdl = tmp_mdl
                except: continue                    
    return best_aic, best_order, best_mdl


def garch_univariate(prc, p=1, o=1, q=1, dist='Normal', bestGARCH=False, maxp=3, maxq=3, maxo=3):
    """
    Compute a univariate GARCH model from a given principal components matrix
    If bestGARCH = TRUE the function computes the best model in terms of AIC
    """
    if not isinstance(prc, np.ndarray):
        prc = np.array(prc)
        
    var = np.zeros(shape=prc.shape[1])
    
    for i in range(prc.shape[1]):
        
        if not prc[:,i].flags['C_CONTIGUOUS']:
            serie = prc[:,i].copy(order = 'C')
        else:
            serie = prc[:,i]
        # Finding the best model
        if bestGARCH:
            model = best_garch_model(100*serie, maxp=maxp, maxo=maxo, maxq=maxq, dist=dist)[2]
            var[i] = model.forecast().variance.iloc[-1,-1]/10000
        else:
            model = arch_model(100*serie, mean='Zero', vol='GARCH', p=p, o=o, q=q, dist=dist)
            with suppress_stdout(): 
                res = model.fit()
            var[i] = res.forecast().variance.iloc[-1,-1]/10000
    return pd.DataFrame(np.diag(var))

def orthogonal_garch_cov(r, percent_explained=99.0, **kwargs):
    """
    Computes the principal components for have the GARCH model applied for
    estimate its variance and then estimates the covariance matrix
    """
    nrow = r.shape[0]
    ncol = r.shape[1]
    r_std = StandardScaler().fit_transform(r)/np.sqrt(nrow)
    X_T_X = (r_std.T @ r_std)
    eigvals, W = np.linalg.eig(X_T_X)
    if percent_explained <= 1.0 or percent_explained >= 100.0:
        raise ValueError('percent_explained must be between 1 and 100')
    else:
        tot = sum(eigvals)
        var_exp = [(i / tot)*100 for i in sorted(eigvals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        cum_var_exp = pd.DataFrame(cum_var_exp)
        index = cum_var_exp[cum_var_exp[0]>percent_explained].index.tolist()
        m = index[0]
    W_new = W[:,0:m]
    p = r_std @ W_new
    sigma = np.array(r.std(ddof=0)).reshape(ncol,1)
    A = W_new * sigma
    var_mat_garch = garch_univariate(p, **kwargs)
    V = A @ var_mat_garch @ A.T
    return pd.DataFrame(V)
    
def summary_stats(r, w, riskfree_rate=0.03, returns=None):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """

    r = r.dropna()
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    if isinstance(returns, (pd.DataFrame, list, np.ndarray)):
        aenc=np.zeros(w.shape[1])
        aencb=np.zeros(w.shape[1])
        for i in range(w.shape[1]):
            aenc[i] = enc(w.iloc[0,i])
            aencb[i] = encb(returns, w.iloc[0,i])
        aenc = pd.Series(aenc,index=w.columns)
        aencb = pd.Series(aencb,index=w.columns)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
        "Average ENC": aenc,
        "Average ENCB": aencb
    })