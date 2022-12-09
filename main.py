from src.model import MyModel
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split
from src.dataset import MyDataset
from src.trainer import TRAINER
import argparse
import os
import wandb

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES # cuda:0, 0
    
    wandb.init(project=args.project, entity="umeyuu")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset(args.max_len)

    # データセットを分割
    train_size = int(args.rate_train_val * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # データローダー
    train_dataloader = DataLoader(
                train_dataset,  
                # sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
                batch_size = args.batch_size
            )
    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), # 順番にデータを取得してバッチ化
                batch_size = 1
            )

    model = MyModel(max_len=args.max_len)
    # 損失関数の設定(クロスエントロピー誤差)
    criterion = CrossEntropyLoss() #この中でソフトマックス関数と同じ処理をしている
    # 最適化手法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    Tr = TRAINER(model, device, train_dataloader, validation_dataloader, criterion, optimizer, scheduler)

    for epoch in range(1, args.epoch+1):
        train_loss = Tr.train()
        val_loss = Tr.validation()
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'epoch={epoch}')
        torch.save(Tr.model.to('cpu').state_dict(), args.save_path + f'epoch={epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='Predict Satelite Potential')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='MIG-10d45f35-eccc-50c5-a994-7971ed3e6673')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--rate_train_val', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_path', default='save_model/')
    args = parser.parse_args()

    # モデルの保存先がないなら作成する。
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    main(args)