import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import statsmodels.api as sm
import pandas as pd
from numpy import random as rn
import pickle

def z_score(series):
    return (series - series.mean()) / np.std(series)

with open('PriceData_Trading.pkl', 'rb') as f:
    price_trading = pickle.load(f)
with open('PriceData_Train.pkl', 'rb') as f:
    price_training = pickle.load(f)
with open('PriceData_Full.pkl', 'rb') as f:
    price_full = pickle.load(f)
with open('PriceData_TrainExcl2020.pkl', 'rb') as f:
    price_trainexcl = pickle.load(f)
    

P1 = price_trading["PG"]
P2 = price_trading["ULVR"]

#Observe the range of the z-score
#Warning: can't determine the range of z-score by looking at the plot because of the future function
ratio = P1/P2
z_ratio = z_score(ratio)

"""plt.plot(np.array(z_ratio))
plt.plot(1.5 * np.ones(len(z_ratio)))
plt.plot(-1.5 * np.ones(len(z_ratio)))
plt.plot(np.ones(len(z_ratio)))
plt.plot(-np.ones(len(z_ratio)))
plt.plot(2 * np.ones(len(z_ratio)))
plt.plot(-2 * np.ones(len(z_ratio)))
plt.plot(0.5 * np.ones(len(z_ratio)))
plt.plot(-0.5 * np.ones(len(z_ratio)))
plt.plot(np.zeros(len(z_ratio)))
plt.show()"""


#Parameters
ins_window = 5
long_window = 60
fee = 0.0002
trading_volume = 0.3
deviation = 1.5
closing = 0.5
stoploss = 2

#Calculate z-score using rolling window
ins_ratio = ratio.rolling(ins_window).mean()
long_mean = ratio.rolling(long_window).mean()
long_std = ratio.rolling(long_window).std()
z_rolling = (ins_ratio - long_mean) / long_std
#Validate from 2021-03-30
#print(ins_ratio[58:60])
#print(z_rolling[58:61])


def Trade(price1, price2, volume, fee, capital, holding1, holding2):
    #Short 1, Long 2
    trading_capital = capital * volume
    capital -= 2 * trading_capital * fee
    holding1 -= trading_capital / price1
    holding2 += trading_capital / price2
    return capital, holding1, holding2

def Close(price1, price2, fee, capital, holding1, holding2):
    trading_capital = abs(holding1) * price1 + abs(holding2) * price2
    capital -= trading_capital * fee
    holding1 = 0
    holding2 = 0
    return capital, holding1, holding2

#Start trading
holding1 = 0
holding2 = 0
capital = 1
capital_list = [1]
holding1_list = [0]
holding2_list = [0]

for i in range(59, len(z_rolling)):
    #print(i)
    capital = capital + holding1 * (P1[i] - P1[i - 1]) + holding2 * (P2[i] - P2[i - 1])
    if z_rolling[i] > deviation:
        #Short P1, Long P2
        if holding1 == 0 and holding2 == 0:
            capital, holding1, holding2 = Trade(P1[i], P2[i], trading_volume, fee, capital, holding1, holding2)
            
        if holding1 > 0 and holding2 < 0:
            capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)
            capital, holding1, holding2 = Trade(P1[i], P2[i], trading_volume, fee, capital, holding1, holding2)
            
    if z_rolling[i] < -deviation:
        #Long P1, Short P2
        if holding1 == 0 and holding2 == 0:
            capital, holding2, holding1 = Trade(P2[i], P1[i], trading_volume, fee, capital, holding2, holding1)
            
        if holding1 < 0 and holding2 > 0:
            capital, holding2, holding1 = Close(P2[i], P1[i], fee, capital, holding2, holding1)
            capital, holding2, holding1 = Trade(P2[i], P1[i], trading_volume, fee, capital, holding2, holding1)

    if abs(z_rolling[i]) < closing:
        #Close position    
        capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)
    
    if abs(z_rolling[i]) > stoploss:
        #Stop loss
        capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)

    capital_list.append(capital)
    holding1_list.append(holding1)
    holding2_list.append(holding2)
    
    
plt.plot(capital_list)
#plt.plot(holding1_list)
#plt.plot(holding2_list)
plt.show()


