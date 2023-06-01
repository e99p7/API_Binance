import ssl
import time
import json
import urllib
import hmac, hashlib
import requests
import pandas as pd

from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen

from pasta.augment import inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, Adagrad, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LambdaCallback

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


import random
import math
import os
import re
import matplotlib.pyplot as plt
#%matplotlib inline

from IPython.display import clear_output

# Шаг 1: Получение данных с использованием API Binance

url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '1h',
    'limit': 1000  # Максимальное количество свечей
}

response = requests.get(url, params=params)
data = response.json()

# Преобразование данных в Pandas DataFrame и сохранение в CSV

df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                 'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])

df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df.to_csv('binance_data.csv', index=False)

df = pd.read_csv('binance_data.csv')


