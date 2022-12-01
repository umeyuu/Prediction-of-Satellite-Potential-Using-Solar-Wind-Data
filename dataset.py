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

        paths = self.get_file_path()

        df = pd.DataFrame()
        for path in paths:
            df_tmp = self.load_data(path, df)
            df = df_tmp
        # df = df[df.flow_speed != 99999.898438]
        df = df.reset_index(drop=True)
        self.x_time = df.date
        self.df_solar = df.copy()
        df.drop('date', axis=1, inplace=True)

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

    # 太陽風データファイルのパスを取得
    def get_file_path(self):
        file_lis = []
        for year in range(2010, 2015):
            dir = f'DATA/solar_wind/{year}/'
            files = os.listdir(dir)
            tmp = [dir+f for f in files if os.path.isfile(os.path.join(dir, f))]
            tmp = sorted(tmp)
            file_lis.extend(tmp)
        return file_lis
    
    def load_data(self, path, df):
        # cdfファイル読み込み
        cdf_file = cdflib.CDF(path)
        # 説明変数
        X = ['BX_GSE', 'BY_GSM', 'BZ_GSM', 'flow_speed', 'Vx', 'Vy', 'Vz', 'proton_density', 'T', 'Pressure']
        tmp = []
        for x in X:
            tmp.append(cdf_file[x])
        tmp = np.array(tmp)
        # 時間
        epoch = cdflib.cdfepoch.unixtime(cdf_file['Epoch'])
        date = [datetime.utcfromtimestamp(e) for e in epoch]

        df_tmp = pd.DataFrame(tmp.T, columns=X)
        df_tmp['date'] = date

        return pd.concat([df, df_tmp])



dataset = MyDataset(60)
print(dataset[0])
breakpoint()

