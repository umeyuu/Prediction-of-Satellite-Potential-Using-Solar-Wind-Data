from src.model import MyModel
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data import random_split
from src.dataset import MyDataset2
from src.trainer import TRAINER
from src.LossFunction import LDLLoss
import argparse
import os
import wandb
import numpy as np

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES # cuda:0, 0
    
    wandb.init(project=args.project, entity="umeyuu")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset2(args.max_len)

    # データセットを分割
    train_size = int(args.rate_train_val * len(dataset))
    indices = np.arange(len(dataset))
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:])

    # データローダー
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
                batch_size = args.batch_size
            )
    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), # 順番にデータを取得してバッチ化
                batch_size = args.batch_size
            )

    model = MyModel(max_len=args.max_len)
    # 損失関数の設定
    # criterion = CrossEntropyLoss() #この中でソフトマックス関数と同じ処理をしている
    criterion = LDLLoss(dist_size=103, devise=device)
    # 最適化手法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    Tr = TRAINER(model, device, train_dataloader, validation_dataloader, criterion, optimizer, scheduler)

    for epoch in range(1, args.epoch+1):
        train_loss = Tr.train()
        val_loss = Tr.validation()
        auc, accuracy, precision, recall, f1 = Tr.chaek_performance()

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'epoch={epoch}')
        torch.save(Tr.model.to('cpu').state_dict(), args.save_path + f'epoch={epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='Predict Satelite Charge Count')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='MIG-2830f874-62b4-5f63-b39a-c4ae68478be0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=60)
    parser.add_argument('--rate_train_val', type=float, default=0.8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_path', default='save_model/')
    args = parser.parse_args()

    # モデルの保存先がないなら作成する。
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    main(args)