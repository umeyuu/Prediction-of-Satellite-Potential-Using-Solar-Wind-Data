import torch
from torch.utils.data import Dataset
import cdflib
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, paths, window_size):
        '''''
        args
        paths : 読み込みファイルのpathの配列,  window_size : 過去何個のデータをモデルに入力するか
        '''''
        super().__init__()

        df = pd.DataFrame()
        for path in paths:
            df_tmp = self.load_data(path, df)
            df = df_tmp
        self.df = df

        # 正規化処理
        scaler = MinMaxScaler()
        self.train_array = scaler.fit_transform(df)
        self.date = df.index
        self.input_size = len(df)
        self.window_size = window_size
        
    def __len__(self):
        return self.input_size
    
    def __getitem__(self, index):
        return self.train_array[index : index+self.window_size]
    
    def load_data(self, path, df):
        # cdfファイル読み込み
        cdf_file = cdflib.CDF(path)
        X = ['BX_GSE', 'BY_GSM', 'BZ_GSM', 'flow_speed', 'Vx', 'Vy', 'Vz', 'proton_density', 'T', 'Pressure']
        tmp = []
        for x in X:
            tmp.append(cdf_file[x])
        tmp = np.array(tmp)
        # 時間
        epoch = cdflib.cdfepoch.unixtime(cdf_file['Epoch'])
        date = [datetime.utcfromtimestamp(e) for e in epoch]
        df_tmp = pd.DataFrame(tmp.T, columns=X, index=date)
        return pd.concat([df, df_tmp])



path = ['DATA/omni_hro_1min_20100101_v01.cdf', 'DATA/omni_hro_1min_20100201_v01.cdf']
dataset = MyDataset(path, 5)
print(dataset[0])
breakpoint()