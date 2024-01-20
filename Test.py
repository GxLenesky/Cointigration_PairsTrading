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
    

P1 = price_trading["PEP"]
P2 = price_trading["KO"]

#Observe the range of the z-score
ratio = P1/P2
z_ratio = z_score(ratio)

plt.plot(np.array(z_ratio))
plt.plot(1.5 * np.ones(len(z_ratio)))
plt.plot(-1.5 * np.ones(len(z_ratio)))
plt.plot(np.ones(len(z_ratio)))
plt.plot(-np.ones(len(z_ratio)))
plt.plot(2 * np.ones(len(z_ratio)))
plt.plot(-2 * np.ones(len(z_ratio)))
plt.plot(0.5 * np.ones(len(z_ratio)))
plt.plot(-0.5 * np.ones(len(z_ratio)))
plt.plot(np.zeros(len(z_ratio)))
#plt.show()
#Choose 1 sigma as deviation

#Parameters
ins_window = 5
long_window = 60
deviation = 1

#Calculate z-score using rolling window
ins_ratio = ratio.rolling(ins_window).mean()
long_mean = ratio.rolling(long_window).mean()
long_std = ratio.rolling(long_window).std()
z_rolling = (ins_ratio - long_mean) / long_std
#print(ins_ratio[:15])
print(z_rolling[40])






