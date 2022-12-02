import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
        df_sat = pd.read_csv('DATA/dmsp-f16_data.csv', parse_dates=["date"])
        df_sat = df_sat.sort_values('date')
        self.df_sat = df_sat.reset_index(drop=True)
        self.target_time = self.df_sat.date

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
        ind = self.x_time[self.x_time == time].index.item() + 1
        x = self.train_array[ind-self.window_size : ind]

        return torch.tensor(x), torch.tensor(y)
    
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


train_data = MyDataset(60)
trainloader = DataLoader(train_data, batch_size = 8, shuffle = True)

for x, y in trainloader:
    print(x, y)
    breakpoint()
