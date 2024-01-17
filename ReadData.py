import numpy as np
import pandas as pd
import pickle

price_trading = pd.read_csv('PriceData_Trading.csv')
price_training = pd.read_csv('PriceData_Train.csv')
price_full = pd.read_csv('PriceData_Full.csv')
price_trainexcl = pd.read_csv('PriceData_TrainExcl2020.csv')

with open('PriceData_Trading.pkl', 'wb') as f:
    pickle.dump(price_trading, f)
with open('PriceData_Train.pkl', 'wb') as f:
    pickle.dump(price_training, f)
with open('PriceData_Full.pkl', 'wb') as f:
    pickle.dump(price_full, f)
with open('PriceData_TrainExcl2020.pkl', 'wb') as f:
    pickle.dump(price_trainexcl, f)