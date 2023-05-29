import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timezone, timedelta

import os
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 10000)
pd.set_option('display.width', 10000)

"""
1. 预报模式每隔12小时输出未来预测结果 输出时间是每天的0点和12点
2. 时间间隔是000 003 一直到240
3. 预报模式内的数据包含经纬度和气象观测数据 按需使用
4. 观测站要按照站点 进行数据预处理
5. 对模式数据的预处理 F_time, time_gap, 样本 其中样本应该是array 和经纬度对应了
Question
q0:模式预报给出的F_time和站点时间 不会一一对应，如何解决？
a0:本身站点数据是按照自然小时进行聚合的指标
"""

# long = 721 lat = 561
lon_range = (121.875, 122.125)
lat_range = (29.25, 29.625)
delta_degree = 0.125
lon_start = 60.0
lat_start = 60.0
rad = np.pi / 180.0
lon_index_range = range(int((lon_range[0] - lon_start) / delta_degree),
                        int((lon_range[1] - lon_start) / delta_degree) + 1)
lat_index_range = range(int((-lat_range[1] + lat_start) / delta_degree),
                        int((-lat_range[0] + lat_start) / delta_degree) + 1)
target_stations = ['K2931', 'K2942', 'K2962']

"K2913 lat:29.50 log:122.0"
"K2942 lat:29.50 log:122.0"
"K2962 lat:29.50 log:122.0"
"Attention! lat is descresing!  lat:29.50 lat_idx=1, log:122.0 log_idx = 1"
"so after reshape it is 1*3+1 = 4"

NC_FILE_PATH = ['/Volumes/zcy单位备份/sailing/nc_ecf/2021.09'
    , '/Volumes/zcy单位备份/sailing/nc_ecf/2021.10'
    , '/Volumes/zcy单位备份/sailing/nc_ecf/2022.09'
    , '/Volumes/zcy单位备份/sailing/nc_ecf/2022.10']

FILE_NAME_RE = r'(ecfine)\.(.*)\.(\d+)\.(.*)\.(nc)'

TARGET_STATIONS_FILES_PREFIX = '/Volumes/zcy单位备份/sailing/'
TARGET_STATIONS_FILES = ['9月亚帆赛三个站点的风场数据.csv',
                         '10月亚帆赛三个站点的风场数据.csv',
                         '21年10月亚帆赛三个站点的风场数据.csv',
                         '21年9月亚帆赛三个站点的风场数据.csv']


def ecfProcess(file_path, f):
    fpath = os.path.join(file_path, f)
    fn = nc.Dataset(fpath)
    print(f, fpath, fn.variables.keys())
    fg310 = fn.variables['fg310'][:, lat_index_range.start:lat_index_range.stop,lon_index_range.start:lon_index_range.stop]
    t2 = fn.variables['t2'][:, lat_index_range.start:lat_index_range.stop,lon_index_range.start:lon_index_range.stop]
    u10 = fn.variables['u10'][:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    v10 = fn.variables['v10'][:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    u100 = fn.variables['u10'][:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    v100 = fn.variables['v10'][:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    return [re.findall(FILE_NAME_RE, f)[0][2]
            , re.findall(FILE_NAME_RE, f)[0][3]
            , np.concatenate([np.array(fg310).reshape(-1),
                              np.array(t2).reshape(-1),
                              np.array(u10).reshape(-1),
                              np.array(v10).reshape(-1),
                              np.array(u100).reshape(-1),
                              np.array(v100).reshape(-1)], axis=0).tolist()]


# 进行时区转换和模式预报对应
#
def observeTimeProcess(x):
    dt_str = str(x) + '+0800'
    dt_obj = datetime.strptime(dt_str, '%y%m%d%H%z')
    tz_utc_8 = timezone(timedelta(hours=8))
    tz_utc = timezone(timedelta(hours=0))
    return 'F' + dt_obj.astimezone(tz_utc).strftime("%Y%m%d%H")


file_names_collect = list(filter(
    lambda x: 72 >= int(re.findall(FILE_NAME_RE, x[1])[0][2]) >= 3,
    [(fpathe, f) for fname in NC_FILE_PATH for fpathe, dirs, fs in os.walk(fname) for f in fs]))
print(file_names_collect)
p = [ecfProcess(tmp[0], tmp[1]) for tmp in file_names_collect]
data_df = pd.DataFrame(p, columns=['gap', 'F_time', 'input'])
data_df.to_pickle("data.pkl")


# pd process


def process_station():
    df = pd.concat([pd.read_csv(TARGET_STATIONS_FILES_PREFIX + x) for x in TARGET_STATIONS_FILES])[
        ['StationNum', 'ObservTimes', 'WindDirect10', 'WindVelocity10', 'ExMaxWindV']]
    df['F_time'] = df['ObservTimes'].apply(lambda x: observeTimeProcess(x))
    # df['u10'] = df.apply(lambda x: -x['WindVelocity10'] * np.sin(x['WindDirect10'] * rad), axis=1) * 0.1
    # df['v10'] = df.apply(lambda x: -x['WindVelocity10'] * np.cos(x['WindDirect10'] * rad), axis=1) * 0.1
    df['ExMaxWindV'] = df['ExMaxWindV'] * 0.1
    df = pd.DataFrame(df, columns=['StationNum', 'F_time', 'ExMaxWindV']) \
        .groupby(['F_time'], sort=False).aggregate(list)

    df['label'] = df.apply(lambda x: sorted(list(zip(x.StationNum, x.ExMaxWindV))
                                            , key=lambda t: t[0], reverse=False), axis=1)
    return df[['label']]


df = process_station().reset_index()
df.to_pickle("station.pkl")
