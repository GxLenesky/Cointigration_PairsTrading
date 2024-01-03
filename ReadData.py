import numpy as np
import pandas as pd
import pickle

PairsPrice = np.array(pd.read_csv('PriceData_Trading.csv', usecols = ["BAC", "JPM"]))

with open('PairsPrice.pkl', 'wb') as f:
    pickle.dump(PairsPrice, f)