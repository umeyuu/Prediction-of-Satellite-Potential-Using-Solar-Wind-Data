import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from typing import Tuple
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
        # df_charge = self.SearchBalancedData(df_charge, num=1000, col='five_label') # 均衡データになるようにラベルを振り直す
        self.df_charge = self.undersampling(df_charge, num=1000, col='five_label') # アンダーサンプリング

        self.time = self.df_charge['date'].values
        self.charge_count = self.df_charge['five_label'].values
        # 緯度の平均と標準偏差を取得
        self.lat_m = self.df_charge.lat.abs().mean()
        self.lat_s = self.df_charge.lat.abs().std()

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
        # 緯度経度を正規化
        lat = (lat - self.lat_m) / self.lat_s
        lon = math.sin(lon/2)
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
    def undersampling(self, df, num=1000, col: str ='ten_label'):

        charge_counts = df[col].unique()
        df_new = pd.DataFrame()

        for count in charge_counts:
            df_tmp = df[df[col] == count]
            if len(df_tmp) > num:
                df_tmp = df_tmp.sample(n=num)
                df_new = pd.concat([df_new, df_tmp])
        
        df_new = df_new.sort_values(by='date')
        df_new.reset_index(drop=True, inplace=True)
        
        return df_new
    
    # 均衡データを取得
    def SearchBalancedData(self, df: pd.DataFrame, num: int, col: str = 'ten_label'):
        charge_dict = df[col].value_counts().to_dict()
        charge_dict = sorted(charge_dict.items())
        count = 0
        key_list = []
        value_list = []
        ans = dict()

        for k, v in charge_dict:
            count += v
            value_list.append(v)
            key_list.append(k)
            if count > num:
                weighted_average = np.dot(key_list, value_list) / count
                for k in key_list:
                    ans[k] = weighted_average
                count = 0
                value_list = []
                key_list = []

        weighted_average = np.dot(key_list, value_list) / count
        for k in key_list:
            ans[k] = weighted_average
        
        df[col] = df[col].apply(lambda x: ans[x])
        return df
    




class MyDataset3(Dataset):
    def __init__(self, window_size: int) -> None:

        self.window_size = window_size

        df_charge = pd.read_csv('dmsp-f16.csv', parse_dates=['date'])
        df_solar = pd.read_csv('solar_wind.csv', parse_dates=['date'])
        df_solar = self.normalize_SolarWind(df_solar)
        df = pd.merge(df_charge, df_solar, on='date')
        df.lat = df.lat.abs()
        self.df = df
        self.targetIndex = self.extractTargetIndex()
        

        # self.df.drop(['satellite_id', 'date', 'flow_speed', ], axis=1, inplace=True)
        self.df.drop(['satellite_id', 'date', 'flow_speed',  'AE_INDEX', 'AL_INDEX', 'AU_INDEX'], axis=1, inplace=True)
        
        
    
    def __len__(self) -> int:
        return len(self.targetIndex)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        target_idx = self.targetIndex[idx]

        df_tmp = self.df.iloc[target_idx - self.window_size: target_idx+1]
        # 帯電回数
        charge_count = self.df.iloc[target_idx].charge_count
        charge_count = 0 if charge_count == 0 else 1
        charge_count = torch.tensor(charge_count, dtype=torch.int64)

        # 入力データ
        tgt = self.get_latlon(df_tmp)
        src = df_tmp.drop(['charge_count', 'lat', 'lon'], axis=1)
        src = torch.tensor(src.values, dtype=torch.float32)

        return src, tgt, charge_count
        
    
    def extractTargetIndex(self) -> np.ndarray:
        # オーロラ帯のデータを抽出
        df_tmp = self.df[self.df.lat >= 55][self.window_size+1 :]
        df_tmp.lon = df_tmp.lon.apply(lambda x: x * np.pi / 12)
        df_tmp = df_tmp[(df_tmp.lon <= 2) | (df_tmp.lon >= 4.5)]
        # df_tmp = self.undersampling(df_tmp, num=90)
        df_tmp = self.cutTime(df_tmp)
        df_tmp = self.undersampling_cls(df_tmp)
        self.target_df = df_tmp

        # 緯度の平均と標準偏差を取得
        self.lat_m = df_tmp.lat.mean()
        self.lat_s = df_tmp.lat.std()
        
        return df_tmp.index.values


    # アンダーサンプリング
    def undersampling(self, df, num=1000, col: str ='charge_count'):

        charge_counts = df[col].unique()
        df_new = pd.DataFrame()

        for count in charge_counts:
            df_tmp = df[df[col] == count]
            if len(df_tmp) > num:
                df_tmp = df_tmp.sample(n=num)
                df_new = pd.concat([df_new, df_tmp])
        
        df_new = df_new.sort_values(by='date')
        # df_new.reset_index(drop=True, inplace=True)
        
        return df_new
    
    def undersampling_cls(self, df):
        df_charge = df[df.charge_count >= 1]
        df_nocharge = df[df.charge_count == 0].sample(n=len(df_charge))
        df_new = pd.concat([df_charge, df_nocharge])
        df_new = df_new.sort_values(by='date')

        return df_new
    
    # AE_INDEXが2019年4月以降0であるため、それ以降のデータを削除
    def cutTime(self, df):
        df = df[df.date < '2019-04']
        return df

    
    # 太陽風データを正規化
    def normalize_SolarWind(self, df_solar):
        scaler = MinMaxScaler()
        date = df_solar['date']
        solar = df_solar.drop(['date'], axis=1)
        df = scaler.fit_transform(solar)
        df = pd.DataFrame(df, columns=solar.columns)
        df['date'] = date
        
        return df
    
    # 緯度経度を正規化
    def get_latlon(self, df):
        latlon = df[['lat', 'lon']].values[-1].tolist()

        # 緯度経度をラジアンに変換
        lat = abs(latlon[0])
        lon = latlon[1] * np.pi /12
        # 緯度経度を正規化
        lat = (lat - self.lat_m) / self.lat_s
        lon = math.sin(lon/2)
        tgt = [lat, lon]
        
        return torch.tensor(tgt).to(torch.float32)



class MyDataset4(Dataset):
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        # 帯電データを読み込み
        self.df_charge = self.creatChargeDataFrame()
        self.df_charge.sort_values(by='date', inplace=True)

        # 太陽風データを読み込み
        df_solar = pd.read_csv('solar_wind.csv', parse_dates=['date'])
        self.df_solar = self.normalize_SolarWind(df_solar)
        self.df_solar.set_index('date', inplace=True)

        # 説明変数を選択
        self.df_solar.drop(['flow_speed', ], axis=1, inplace=True)
        # self.df_solar.drop(['flow_speed',  'AE_INDEX', 'AL_INDEX', 'AU_INDEX'], axis=1, inplace=True)

        # 緯度の平均と標準偏差を取得
        self.lat_m = self.df_charge.lat.mean()
        self.lat_s = self.df_charge.lat.std()

    def __len__(self) -> int:
        return len(self.df_charge)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_df = self.df_charge.iloc[idx]
        target_time = target_df.date
        src = self.df_solar[target_time - pd.Timedelta(minutes=self.window_size-1): target_time]
        src = torch.tensor(src.values, dtype=torch.float32)

        tgt = self.getLatLon(target_df)

        # 帯電かどうか
        charge_count = target_df.charge_count
        charge = 0 if charge_count == 0 else 1
        charge = torch.tensor(charge, dtype=torch.int64)

        return src, tgt, charge
    
    # アンダーサンプリング
    def undersampling(self, df: pd.DataFrame) -> pd.DataFrame:
        df_charge = df[df.charge_count >= 1]
        df_nocharge = df[df.charge_count == 0].sample(n=len(df_charge))
        df_new = pd.concat([df_charge, df_nocharge])
        df_new = df_new.sort_values(by='date')

        return df_new
    
    # AE_INDEXが2019年4月以降0であるため、それ以降のデータを削除
    def cutTime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.date < '2019-04']
        return df
    
    # オーロラ帯のデータのみ抽出
    def extractAuroraBelt(self, df: pd.DataFrame) -> pd.DataFrame:
        df.lat = df.lat.abs()
        df.lon = df.lon.apply(lambda x: x * np.pi / 12)
        df = df[df.lat >= 55]
        df = df[(df.lon <= 2) | (df.lon >= 4.5)]

        return df
    
    # 学習データを作成
    def extractTrainData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.extractAuroraBelt(df)
        df = self.cutTime(df)
        df = self.undersampling(df)
        return df
    
    def creatChargeDataFrame(self) -> pd.DataFrame:
        # データの読み込み
        df_charge_f16 = pd.read_csv('dmsp-f16.csv', parse_dates=['date'])
        df_charge_f17 = pd.read_csv('dmsp-f17.csv', parse_dates=['date'])
        df_charge_f18 = pd.read_csv('dmsp-f18.csv', parse_dates=['date'])
        
        # データの前処理
        df_charge_f16 = self.extractTrainData(df_charge_f16)
        df_charge_f17 = self.extractTrainData(df_charge_f17)
        df_charge_f18 = self.extractTrainData(df_charge_f18)

        # データの結合
        df = pd.concat([df_charge_f16, df_charge_f17, df_charge_f18], axis=0)
        df.reset_index(drop=True, inplace=True)
        return df
    
    # 太陽風データを正規化
    def normalize_SolarWind(self, df_solar: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        date = df_solar['date']
        solar = df_solar.drop(['date'], axis=1)
        df = scaler.fit_transform(solar)
        df = pd.DataFrame(df, columns=solar.columns)
        df['date'] = date
        
        return df
    
    # 緯度経度を取得
    def getLatLon(self, ser: pd.Series) -> Tuple[float, float]:
        lat = ser.lat
        lon = ser.lon
        # 緯度経度を正規化
        lat = (lat - self.lat_m) / self.lat_s
        lon = math.sin(lon/2)
        
        latlon = [lat, lon]
        return torch.tensor(latlon).to(torch.float32)