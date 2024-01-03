import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import statsmodels.api as sm
import pandas as pd
from numpy import random as rn
import pickle

with open('PairsPrice.pkl', 'rb') as f:
    PairsPrice = pickle.load(f)
    
# Parameters
T = len(PairsPrice)
# fee = 0.001 (might be a parameter)
window = 252 # window size for rolling regression
t_threshold = -2.5 # t-statistic threshold, to be changed
stock1 = PairsPrice[:, 0]
stock2 = PairsPrice[:, 1]
returns_rate = np.append(np.zeros((1, 2)), (PairsPrice[1:, :] 
    - PairsPrice[:-1, :]) / PairsPrice[:-1, :], axis = 0)

# Start trading
pairs_returns = np.array([])
for t in range(window, T):
    # Estimate the cointegration relationship
    def unit_root(b):
        a = np.average(stock2[t - window:t] - b * stock1[t - window:t])
        fair_value = a + b * stock1[t - window:t]
        diff = np.array(fair_value - stock2[t - window:t])
        diff_diff = diff[1:] - diff[:-1]
        reg = sm.OLS(diff_diff, diff[:-1])
        res = reg.fit()
        return res.params[0]/res.bse[0]
    
    #optimising the cointegration equation parameters
    res1 = opt.minimize(unit_root, stock2[t]/stock1[t], method = 'Nelder-Mead')
    t_opt = res1.fun
    b_opt = float(res1.x)
    a_opt = np.average(stock2[t - window:t] - b_opt * stock1[t - window:t])
    
    fair_value = a_opt + b_opt * stock1[t]
    if t_opt > t_threshold:
        signal = 0
        gross_return = 0
    else:
        signal = np.sign(fair_value - stock2[t])
        gross_return = signal * returns_rate[t][1] - signal * returns_rate[t][0]
    
    pairs_returns = np.append(pairs_returns, gross_return)
    
    if t % 50 == 0:
        print('gross daily return: '+str(round(gross_return * 100,2))+'%')
        print('cumulative net return so far: '+str(round(np.prod(1 + pairs_returns)*100 - 100, 2))+'%')
        print('')
    
    plt.plot(np.append(1,np.cumprod(1 + pairs_returns)))
plt.show()





