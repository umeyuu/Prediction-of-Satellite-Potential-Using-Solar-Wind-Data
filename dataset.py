import torch
from torch.utils.data import Dataset
import cdflib
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self, window_size):
        '''''
        args
        paths : 読み込みファイルのpathの配列,  window_size : 過去何個のデータをモデルに入力するか
        '''''
        super().__init__()

        # 太陽風データを取得
        df = pd.read_csv('DATA/solar_wind/solar_wind_data.csv', parse_dates=["date"], index_col=False)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        # df = df[df.flow_speed != 99999.898438]
        self.x_time = df.date
        self.df_solar = df.copy()
        df.drop('date', axis=1, inplace=True)

        # 衛星データを取得
        df_sat = pd.read_csv('DATA/dmsp-f16/dmsp-f16_data.csv', parse_dates=["date"])
        df_sat = df_sat.sort_values('date')
        self.df_sat = df_sat.reset_index(drop=True)
        self.target_time = self.df_sat.date

        # 正規化処理
        scaler = MinMaxScaler()
        self.train_array = scaler.fit_transform(df)
        
        self.input_size = len(df)
        self.window_size = window_size
        
    def __len__(self):
        return self.input_size
    
    def __getitem__(self, index):
        time = self.target_time[index]
        y = self.df_sat[index:index+1].values
        i = self.x_time[self.x_time == time].index.item() + 1
        x = self.train_array[i-self.window_size : i]

        return np.array(x), y[0][1:]


dataset = MyDataset(60)
x, y = dataset[0]
breakpoint()

