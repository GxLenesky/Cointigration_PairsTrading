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
    

P1 = price_training["C"]
P2 = price_training["WFC"]

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
closing = 1
stoploss = 2

#Calculate z-score using rolling window
ins_ratio = ratio.rolling(ins_window).mean()
long_mean = ratio.rolling(long_window).mean()
long_std = ratio.rolling(long_window).std()
z_rolling = (ins_ratio - long_mean) / long_std
#Validate from 2010-03-30
#print(ins_ratio[58:60])
#print(z_rolling[58:61])


def Trade(price1, price2, volume, fee, capital, holding1, holding2):
    #Short 1, Long 2
    trading_capital = volume
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
capital_list = []
holding1_list = []
holding2_list = []
action_list = [] #0: no action, 1: short P1, long P2, -1: long P1, short P2, 0: take profit, 2: stop loss, -2: no action

for i in range(long_window - 1, len(z_rolling)):
    #print(i)
    capital = capital + holding1 * (P1[i] - P1[i - 1]) + holding2 * (P2[i] - P2[i - 1])
    if z_rolling[i] > deviation and z_rolling[i] <= stoploss:
        #Short P1, Long P2
        action_list.append(1)
        if holding1 == 0 and holding2 == 0:
            capital, holding1, holding2 = Trade(P1[i], P2[i], trading_volume, fee, capital, holding1, holding2)
            
        if holding1 > 0 and holding2 < 0:
            capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)
            capital, holding1, holding2 = Trade(P1[i], P2[i], trading_volume, fee, capital, holding1, holding2)
            
    elif z_rolling[i] < -deviation and z_rolling[i] >= -stoploss:
        #Long P1, Short P2
        action_list.append(-1)
        if holding1 == 0 and holding2 == 0:
            capital, holding2, holding1 = Trade(P2[i], P1[i], trading_volume, fee, capital, holding2, holding1)
            
        elif holding1 < 0 and holding2 > 0:
            capital, holding2, holding1 = Close(P2[i], P1[i], fee, capital, holding2, holding1)
            capital, holding2, holding1 = Trade(P2[i], P1[i], trading_volume, fee, capital, holding2, holding1)

    elif abs(z_rolling[i]) <= closing:
        #Close position 
        action_list.append(0)   
        capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)
    
    elif abs(z_rolling[i]) > stoploss:
        #Stop loss
        action_list.append(2)
        capital, holding1, holding2 = Close(P1[i], P2[i], fee, capital, holding1, holding2)
    else:
        #No action
        action_list.append(-2)
    
    capital_list.append(capital)
    holding1_list.append(holding1)
    holding2_list.append(holding2)
    
    
plt.plot(capital_list[0:1000])
for i in range(len(action_list[0:1000])):
    if action_list[i] == 1:
        plt.plot(i, capital_list[i], 'ro')  
    if action_list[i] == -1:
        plt.plot(i, capital_list[i], 'go') 
    if action_list[i] == 0:
        plt.plot(i, capital_list[i], 'bo')
    if action_list[i] == 2:
        plt.plot(i, capital_list[i], 'yo')
    if action_list[i] == -2:
        plt.plot(i, capital_list[i], 'ko')
#plt.plot(holding1_list)
#plt.plot(holding2_list)
#plt.show()

cover_front = [0] * 59
complete_action = cover_front + action_list
complete_holding1 = cover_front + holding1_list
complete_holding2 = cover_front + holding2_list
complete_capital = cover_front + capital_list

"""print(len(complete_action), len(complete_holding1), 
    len(complete_holding2), len(complete_capital), len(price_training["Date"]),
    len(P1), len(P2), len(ratio), len(ins_ratio), len(long_mean), len(long_std), len(z_rolling))"""

Input_and_Action = pd.DataFrame({"date": price_training["Date"],
    "C": P1, "WFC": P2, "ratio": ratio, "ins_ratio": ins_ratio,
    "long_mean": long_mean, "long_std": long_std, "z_score": z_rolling, 
    "action": complete_action, "capital": complete_capital,
    "holding1": complete_holding1, "holding2": complete_holding2})

Input_and_Action.to_csv("C_WFC_training.csv", index = False)


