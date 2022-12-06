import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import math


class MyDataset(Dataset):
    def __init__(self, window_size):
        '''''
        args
        paths : 読み込みファイルのpathの配列,  window_size : 過去何個のデータをモデルに入力するか
        '''''
        super().__init__()

        # 太陽風データを取得
        df = pd.read_csv('DATA/solar_wind_data.csv', parse_dates=["date"], index_col=False)
        self.x_time = df.date
        df.drop(['Unnamed: 0', 'date'], axis=1, inplace=True)
        df = self.process_nan(df) # 欠損値処理
        self.df_solar = df.copy()
       
        # 衛星データを取得
        df_sat = pd.DataFrame()
        for i in range(16, 18):
            tmp = pd.read_csv(f'DATA/dmsp-f{i}/dmsp-f{i}_only_aurora_area.csv', parse_dates=["date"])
            df_sat = pd.concat([df_sat, tmp])

        df_sat = df_sat.sort_values('date')
        self.df_sat = df_sat.reset_index(drop=True)
        self.target_time = self.df_sat.date

        self.lat_m = df_sat.lat.mean()
        self.lat_s = df_sat.lat.std()

        # 正規化処理
        scaler = MinMaxScaler()
        self.train_array = scaler.fit_transform(df)
        
        self.input_size = len(df_sat)
        self.window_size = window_size
        
    def __len__(self):
        return self.input_size
    
    def __getitem__(self, index):
        time = self.target_time[index]
        y = self.df_sat[index:index+1].values
        y = y[0][1:].tolist()
        tgt = y[:-1]
        tgt = self.norm_latlon(tgt)
        ans = y[-1]
        if ans > 0:
            ans = 1

        ind = self.x_time[self.x_time == time].index.item() + 1
        src = self.train_array[ind-self.window_size : ind]

        return torch.tensor(src).to(torch.float32), torch.tensor(tgt).to(torch.float32), torch.tensor(ans).long()
    
    def process_nan(self, df):
        X = df.columns
        V = [9999.99, 9999.99, 9999.99, 99999.9, 99999.9, 99999.9, 99999.9, 999.99, 9999999.0, 99.99]
        for x, v in zip(X, V):
            df[x].replace(v, np.nan, inplace=True)
        # 欠損値があったかどうかの特徴量を付け加える
        flag_nan = lambda x: 1 if x else 0
        df['nan'] = df.isna().any(axis=1).apply(flag_nan)
        df = df.interpolate()
        return df

    #  緯度経度の正規化
    def norm_latlon(self, latlon):
        lat = latlon[0]
        lon = latlon[1]
        # lat = math.sin(lat*(math.pi/180))
        lat = (lat - self.lat_m) / self.lat_s
        lon = math.sin(lon/2)
        out = [lat, lon]
        return out






# a = torch.tensor([1, 1])
# b = torch.tensor([1, 1])
# a = a.to(device)
# b = b.to(device)
# print(a, b)