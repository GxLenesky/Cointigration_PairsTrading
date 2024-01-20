import numpy as np
import pandas as pd
import pickle

price_trading = pd.read_csv('PriceData_Trading.csv', index_col = "Date")
price_training = pd.read_csv('PriceData_Train.csv', index_col = "Date")
price_full = pd.read_csv('PriceData_Full.csv', index_col = "Date")
price_trainexcl = pd.read_csv('PriceData_TrainExcl2020.csv', index_col = "Date")

"""with open('PriceData_Trading.pkl', 'wb') as f:
    pickle.dump(price_trading, f)
with open('PriceData_Train.pkl', 'wb') as f:
    pickle.dump(price_training, f)
with open('PriceData_Full.pkl', 'wb') as f:
    pickle.dump(price_full, f)
with open('PriceData_TrainExcl2020.pkl', 'wb') as f:
    pickle.dump(price_trainexcl, f)"""

#print(np.array(price_trading["TGT"]["2021-01-28" : "2021-02-12"]))
