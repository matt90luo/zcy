import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None

#TARGET_STATIONS = ['K2931', 'K2942', 'K2962']
TARGET_STATIONS = ['K2962']

COLUMNS = [y + "_" + x for x in ['prediction', 'observing', 'ecf'] for y in TARGET_STATIONS]

df = pd.read_pickle('plot.pkl')
df = df[df['K2962_ecf'] > 0]

res = pd.DataFrame(columns=[])

# res['K2931_rmse_clibration'] = df.groupby(['gap']).apply(lambda x: np.sqrt(mean_squared_error(x.K2931_prediction,
#                                                                                               x.K2931_observing
#                                                                                               )))
# res['K2931_rmse_ecf'] = df.groupby(['gap']).apply(
#     lambda x: np.sqrt(mean_squared_error(x['K2931_ecf'], x['K2931_observing'])))
#
# res['K2942_rmse_clibration'] = df.groupby(['gap']).apply(
#     lambda x: np.sqrt(mean_squared_error(x['K2942_prediction'], x['K2942_observing'])))
# res['K2942_rmse_ecf'] = df.groupby(['gap']).apply(
#     lambda x: np.sqrt(mean_squared_error(x['K2942_ecf'], x['K2942_observing'])))


res['K2962_rmse_clibration'] = df.groupby(['gap', 'exp']).apply(
    lambda x: np.sqrt(mean_squared_error(x['K2962_prediction'], x['K2962_observing'])))
res['K2962_rmse_ecf'] = df.groupby(['gap', 'exp']).apply(
    lambda x: np.sqrt(mean_squared_error(x['K2962_ecf'], x['K2962_observing'])))

print(res.head(10000))

res = res.reset_index()
# print(df[df['gap'] <= 72].head(4))

df.to_excel('sailing_plot_details.xlsx', index=False)
res.to_excel('sailing_plot.xlsx', index=False)
