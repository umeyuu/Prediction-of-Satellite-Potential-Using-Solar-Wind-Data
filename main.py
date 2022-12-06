from model import MyModel
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from dataset import MyDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-12dd119f-7dc6-58d8-8d05-9121af063ba6' # cuda:0, 1
# os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-10d45f35-eccc-50c5-a994-7971ed3e6673' # cuda:0, 0
import wandb
wandb.init(project="Predict Satelite Potential", entity="umeyuu")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = MyDataset(60)
trainloader = DataLoader(train_data, batch_size = 64, shuffle = True)
# breakpoint()
model = MyModel()
# 損失関数の設定(クロスエントロピー誤差)
criterion = CrossEntropyLoss() #この中でソフトマックス関数と同じ処理をしている
# 最適化手法の選択(Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model = model.to(device)
criterion = criterion.to(device)
model.train()
for epoch in range(1, 11):
    train_loss = 0
    
    for i, (src, tgt, ans) in enumerate(trainloader):
        src = src.to(device)
        tgt = tgt.to(device)
        ans = ans.to(device)

        out = model(src, tgt)
        loss = criterion(out, ans)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= (i+1)
    wandb.log({'epoch': epoch, 'train_loss': train_loss})
    print(train_loss, i+1)