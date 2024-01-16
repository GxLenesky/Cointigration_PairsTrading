import numpy as np
import pandas as pd
import pickle

PairsPrice = pd.read_csv('PriceData_Trading.csv', usecols = ["BAC", "JPM"])

print(PairsPrice["BAC"][0:10])