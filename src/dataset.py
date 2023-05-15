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
        for i in range(16, 19):
            tmp = pd.read_csv(f'DATA/occure_aurora/1num/dmsp-f{i}_data.csv', parse_dates=["date"])
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
    



class MyDataset2(Dataset):
    def __init__(self, window_size) -> None:
        super().__init__()
        # period : 何分間のデータを取得するか
        self.window_size = window_size

        # 太陽風データを取得
        df_solar = pd.read_csv('solar_wind.csv', parse_dates=["date"], index_col=False)
        self.df_solar = self.normalize_solar(df_solar) # 正規化処理

        # 衛星データを取得
        df_charge = pd.read_csv('charge_kai.csv', parse_dates=['date'], index_col=False)
        df_charge = df_charge.dropna(how='any').reset_index(drop=True) # 欠損値処理
        df_charge['five_label'] = df_charge.apply(self.sum_five, axis=1)
        df_charge['ten_label'] = df_charge.apply(self.sum_ten, axis=1)
        self.df_charge = self.undersampling(df_charge) # アンダーサンプリング

        self.time = self.df_charge['date'].values
        self.charge_count = self.df_charge['ten_label'].values

    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, index):
        # 時間を取得
        time = self.time[index]

        # 時間に対応するデータを取得
        charge = self.df_charge[self.df_charge['date'] == time]
        solar = self.df_solar[(self.df_solar['date'] > time - pd.Timedelta(f'{self.window_size}min')) & (self.df_solar['date'] <= time)]

        # src, tgt, charge_countを取得
        solar = solar.drop(['date'], axis=1)
        solar = solar.values
        src = torch.tensor(solar, dtype=torch.float32)

        charge = charge.drop(['date'], axis=1)
        tgt = self.get_latlon(charge)

        charge_count = self.charge_count[index]
        charge_count = torch.tensor([charge_count], dtype=torch.float32)
        
        return src, tgt, charge_count
    
    def sum_ten(self, df):
        columns = ['charge_count', 'one_minute_after', 'two_minute_after', 'three_minute_after','four_minute_after', 'five_minute_after', 
                'six_minute_after', 'seven_minute_after', 'eight_minute_after', 'nine_minute_after', 'ten_minute_after']
        ans = 0
        for col in columns:
            ans += df[col]
        return ans

    def sum_five(self, df):
        columns = ['charge_count', 'one_minute_after', 'two_minute_after', 'three_minute_after','four_minute_after', 'five_minute_after']
        ans = 0
        for col in columns:
            ans += df[col]
        return ans
    
    def get_latlon(self, df):
        latlon = df[['lat', 'lon']].values[0].tolist()

        # 緯度経度をラジアンに変換
        lat = abs(latlon[0])
        lon = latlon[1] * np.pi /12
        tgt = [lat, lon]
        
        return torch.tensor(tgt).to(torch.float32)
    
    # 太陽風データを正規化
    def normalize_solar(self, df_solar):
        scaler = MinMaxScaler()
        date = df_solar['date']
        solar = df_solar.drop(['date'], axis=1)
        df = scaler.fit_transform(solar)
        df = pd.DataFrame(df, columns=solar.columns)
        df['date'] = date
        
        return df
    
    # アンダーサンプリング
    def undersampling(self, df, num=1000):

        charge_counts = df.ten_label.unique()
        df_new = pd.DataFrame()

        for count in charge_counts:
            df_tmp = df[df['ten_label'] == count]
            if len(df_tmp) > num:
                df_tmp = df_tmp.sample(n=num)
            df_new = pd.concat([df_new, df_tmp])
        
        df_new = df_new.sort_values(by='date')
        df_new.reset_index(drop=True, inplace=True)
        
        return df_new
