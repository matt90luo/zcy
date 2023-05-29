from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 10000)
pd.set_option('display.width', 10000)

def insert_station_time(t, g):
    d = datetime.strptime(t, "F%Y%m%d%H")
    dt = timedelta(hours=int(g))
    return (d - dt).strftime("F%Y%m%d%H")

"尝试去生成一份包含上一时刻的样本数据, F_time - Gap 的时间"

data_df = pd.read_pickle("data.pkl")
df = pd.read_pickle('station.pkl')

print(data_df.head(2))
print(df.head(2))

data_tmp = data_df.merge(df, on="F_time", how='inner')
#print(data_tmp.head(2))
data_tmp['F_time'] = data_tmp.apply(lambda x: insert_station_time(x.F_time, x.gap), axis = 1)
data = data_tmp.merge(df, on="F_time", how='inner')
data['label'] = data.apply(lambda y: [list(map(lambda a: a[1], y.label_x))[-1]], axis=1)
data['extra_station'] = data.apply(lambda y: [list(map(lambda a: a[1], y.label_y))[-1]], axis=1)
data['input'] = data.apply(lambda y: y.input + y.extra_station, axis = 1)
print(data[['gap', 'input', 'label']].head(4))
print(data.columns.tolist())







gap_0 = '006'

# date_1 = datetime.strptime("F2021090103", "F%Y%m%d%H")

print(insert_station_time('F2021090106', '006'))

