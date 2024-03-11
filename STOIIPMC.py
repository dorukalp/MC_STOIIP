# -*- coding: utf-8 -*-
# """
# Created on Sat Feb 24 15:46:53 2024

# @author: Doruk Alp, PhD.
# 2024/03/11, METU-NCC

# Tavory answer @
# https://stackoverflow.com/questions/50294715/monte-carlo-analysis-python-oil-and-gas-volumetrics
# https://github.com/rghalayini/oil_volume_calculator/blob/main/probabilistic_oil_volume_calculator.py
# https://www.instructables.com/How-to-Roll-a-Dice-Using-Python/


# # import os
# # os.chdir(r"C:\Users\labuser.I104-\Downloads\STOIIP_MC.py")
# # os.getcwd()

# """
# *****************************************************************************
# %% LIBRARIES: 
# *****************************************************************************

import random # built-in module

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec



#import scipy.stats as st 
# vs from scipy import stats as st
from scipy.stats import qmc, norm, truncnorm, mode, gaussian_kde

# from statsmodels.distributions.empirical_distribution import ECDF

import seaborn as sns
sns.set_theme(style="ticks")

# import pandas as pd
    
# from icecream import ic

# *****************************************************************************  
# %% main fn.: 
# *****************************************************************************
def main(data):
    """
    num. of dice = # of samples = num. of wells
    target =  avg. of dice values per roll.
    do a MCS, i.e. collect samples (target values) MM times, = num. of rolls
    plot histogram of result of MCS.
    then you can plot results for 1 die or N dice, per your choice.

    Returns
    -------
    None.

    """
    
    stats = np.mean(data), np.percentile(data,10), \
                           np.percentile(data,50), \
                           np.percentile(data,90)
        
    #ic(stats)
    
    nbin = 11
    hist, bin_edges = np.histogram(data, bins=nbin)
        
    plt.hist(data, bins=nbin, edgecolor='black')  # Plot the histogram
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Data')
    plt.draw()


# *****************************************************************************  
# %% main fn.: 
# *****************************************************************************
if __name__ == '__main__':

    # fix seed to get same results each time, on any pc.
    sseed = 9

    random.seed(sseed)  
    
    xRNG = np.random.default_rng(sseed)
    
    np.random.seed(sseed) # supposed to work for scipy too!?
    
    n = int(1E5)
    
    # with built-in random module:
    Area_rnd = [random.triangular(100.0, 300.0, 200.0) for i in range(n)]

    # with numpy:
    Area_np = xRNG.triangular(100, 200, 300, n)

    # with scipy.stats:
    Area_sci = norm(200,50).rvs(n)
    
    Area = Area_sci
    
    # %%     
    h_npB = xRNG.beta(100, 25, size=n) * 100.0

    # %% plot1: Plot the distribution of volumes and their frequency
    plot1=sns.displot(h_npB, kde=True) #, color='skyblue', #hist_kws={"linewidth": 15,'alpha':1})
    plot1.set(xlabel='Normal Distribution', ylabel='Frequency')
    plt.show()

    # %%    
    # with scipy.stats:    
    h = norm(100,25).rvs(n)


    NTG = norm(.85,.2).rvs(n)
    POR = norm(.32,.02).rvs(n)
    Swi = norm(.15,.03).rvs(n)
    Boi = norm(.0024,.0001).rvs(n)

    
    cf = 1.0 # conversion factor
    
    stoiip = Area * h * NTG * POR * (1-Swi)/Boi * cf
    
    #ic(stoiip)


    # %% plot1: Plot the distribution of volumes and their frequency
    plot1=sns.displot(stoiip, kde=True) #, color='skyblue', #hist_kws={"linewidth": 15,'alpha':1})
    plot1.set(xlabel='Normal Distribution', ylabel='Frequency')
    plt.show()
    
    # %%plot2: plot a cumulative distribution of volume and their probability
    plot2=sns.distplot(stoiip, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    
    plot2.set(xlabel='STOIIP mmbbl', ylabel='probability')
    plot2.axhline(y=0.9, label='P10', color="red")
    plot2.axhline(y=0.5, label='P50', color="red")
    plot2.axhline(y=0.1, label='P90', color="red")
    plt.show()
    
    # %% to calculate the P90, P50 and P10 values (90%, 50% and 10% probabilities to have the specified volume):
    stoiip_sorted=np.sort(stoiip)
    x10=int(n/10*9)
    x50=int(n/2)
    x90=int(n/10)
    
    p10=stoiip_sorted[x10]
    p50=stoiip_sorted[x50]
    p90=stoiip_sorted[x90]



    #main()
    raise SystemExit # also removes vars from mem.?     
        
        
        
        
        
        
        
        
        
        