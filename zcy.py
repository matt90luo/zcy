import math

import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timezone, timedelta

from datetime import time


import os
import re

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',10000)




#目标时间范围 9月16号到10月15日
#文件名格式 ecfine.I2022093000.048.F2022100200.nc
example_str = "ecfine.I2022093000.048.F2022100200.nc"
example_re = r'(ecfine)\.(.*)\.(\d+)\.(.*)\.(nc)'
example_re.find(example_str)
print(re.findall(example_re,example_str))
print(re.findall(example_re,example_str)[0][2])
print('.'.join(re.findall(example_re,example_str)[0]))


nc_file_path = '/Volumes/zcy单位备份/sailing/nc_ecf/2021.10'
# for fpathe,dirs,fs in os.walk(nc_file_path):
#   for f in fs:
#     print(os.path.join(fpathe,f))




# long = 721 lat = 561
lon_range = (121.875, 122.125)
lat_range = (29.25, 29.625)
delta_degree = 0.125
lon_start = 60.0
lat_start = 60.0
rad = np.pi/180.0
target_stations = ['K2931', 'K2942', 'K2962']

target_stations_files_prefix = '/Volumes/zcy单位备份/sailing/'
target_stations_files = ['9月亚帆赛三个站点的风场数据.csv',
                         '10月亚帆赛三个站点的风场数据.csv',
                         '21年10月亚帆赛三个站点的风场数据.csv',
                         '21年9月亚帆赛三个站点的风场数据.csv']

file_name_re = r'(ecfine)\.(.*)\.(\d+)\.(.*)\.(nc)'

lon_index_range = range(int((lon_range[0]-lon_start)/delta_degree), int((lon_range[1]-lon_start)/delta_degree) + 1)
lat_index_range = range(int((-lat_range[1]+lat_start)/delta_degree), int((-lat_range[0]+lat_start)/delta_degree) + 1)


def ecfProcess(file_path, f):
    fpath = os.path.join(file_path, f)
    fn = nc.Dataset(fpath)
    fg310 = fn.variables['fg310']
    # u10 = fn.variables['u10']
    # v10 = fn.variables['v10']
    # U = u10[:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    # V = v10[:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
    # X = np.concatenate((U, V), axis=0)
    # return X
    return fg310

def observeTimeProcess(x):
    dt_str = str(x) + '+0800'
    dt_obj = datetime.strptime(dt_str, '%y%m%d%H%z')
    tz_utc_8 = timezone(timedelta(hours=8))
    tz_utc = timezone(timedelta(hours=0))
    return 'F' + dt_obj.astimezone(tz_utc).strftime("%Y%m%d%H")


# IMPORTANT
p = [[re.findall(example_re,example_str)[0][2], ecfProcess(fpathe, f)] for fpathe,dirs,fs in os.walk(nc_file_path) for f in fs]

print("**************")
print(type(p), len(p), type(p[0][0]), type(p[0][1]))
print("**************")


#file_path = '/Volumes/zcy单位备份/sailing/nc_ecf/2022.09/2022093012/ecfine.I2022093012.057.F2022100221.nc'
file_path = '/Volumes/zcy单位备份/sailing/nc_ecf/2022.10/2022100100/ecfine.I2022100100.075.F2022100403.nc'
fn = nc.Dataset(file_path)
print(fn.variables.keys())
print(fn.variables)
print(lon_index_range)
print(lat_index_range)
print([*lon_index_range])
print([*lat_index_range])
fg310 = fn.variables['fg310']
# u10 = fn.variables['u10']
# v10 = fn.variables['v10']
FG = fg310[:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
# U = u10[:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]
# V = v10[:, lat_index_range.start:lat_index_range.stop, lon_index_range.start:lon_index_range.stop]





# process station ExMaxWindV 取过去三小时的极大值
df = pd.concat([pd.read_csv(target_stations_files_prefix+x) for x in target_stations_files])[['StationNum', 'ObservTimes', 'WindDirect10', 'WindVelocity10', 'ExMaxWindV']]
df['F_time'] = df['ObservTimes'].apply(lambda x: observeTimeProcess(x))
df['u10'] = df.apply(lambda x: -x['WindVelocity10'] * np.sin(x['WindDirect10'] * rad), axis=1)*0.1
df['v10'] = df.apply(lambda x: -x['WindVelocity10'] * np.cos(x['WindDirect10'] * rad), axis=1)*0.1
df['ExMaxWindV'] = df['ExMaxWindV']*0.1
print(df.head(2))

df = df.sort_values(['StationNum','ObservTimes'],ascending=False).groupby('StationNum')

print(df.head(4))

#window analysis Important
# df['tmp'] = df.apply(lambda x: (x['StationNum'], x['ExMaxWindV']), axis=1)
# df = df.groupby('ObservTimes').agg({'tmp': lambda x: list(x)})
# df['tmp'] = df.apply(lambda x: sorted(x['tmp']), axis=1)
# df['fg310'] = df['tmp'].apply(lambda x: [ i[1] for i in x])
# df['stationNum'] = df['tmp'].apply(lambda x: [ i[0] for i in x])
# df = df[['fg310', 'stationNum']]
# print(df.head(2))

#dt_str = '27/10/20 05:23:20'
dt_str = '21093021+0800'
dt_obj = datetime.strptime(dt_str, '%y%m%d%H%z')
tz_utc_8 = timezone(timedelta(hours=8))
tz_utc = timezone(timedelta(hours=0))

print("The type of the date is now",  type(dt_obj))
print("The date is", dt_obj)
print("The date is as UTC+8", dt_obj.astimezone(tz_utc_8).strftime("%Y%m%d%H%M"))
print("The date is as UTC", dt_obj.astimezone(tz_utc).strftime("%Y%m%d%H%M"))
ts = dt_obj.timestamp()
print(ts)
print('tz_utc_8 ', datetime.fromtimestamp(ts, tz_utc_8))
print('tz_utc ', datetime.fromtimestamp(ts, tz_utc))

def my_func(a):
    return  (a[0] + a[-1]) * 0.5

b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(np.apply_along_axis(my_func, 0, b))
print(np.apply_along_axis(my_func, 1, b))


# stid  = fn.variables['stid']
# stid  = np.apply_along_axis(lambda x: x.tobytes().decode("utf-8"), 1, stid[:].data)
# df = pd.DataFrame( { 'lon'  : lon[:],
#                      'lat'  : lat[:],
#                      'u10': u10[:,0], # 必须是1维
#                      'v10': v10[:,0],
#                    } )



