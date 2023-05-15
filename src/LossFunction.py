import math
import torch
from torch import nn


class LDLLoss(nn.Module):
    """
    Label Distribution Learning Loss
    """

    def __init__(self, dist_size, delta: float = 1e-6, devise: str = 'cuda'):
        """
        損失関数の初期設定

        Args:
            dist_size (int): 確率分布のサイズ
            delta (float): log(0)を防ぐための微少数
        """
        super(LDLLoss, self).__init__()
        self.dist_size = dist_size
        self.delta = delta
        self.device = devise

    def forward(self, P, y) -> torch.tensor:
        """
        損失の計算

        Args:
            P (torch.tensor(batch_size, self.dist_size)): 予測確率分布
            y (torch.tensor(batch_size)): 正解ラベル
        """
        # 正解クラス y を y を中心とした正規分布 Y に変換
        Y = self.norm_dist(y)
        # Cross Entropy
        loss = -Y * torch.log(P + self.delta)
        
        return torch.mean(loss)

    def norm_dist(self, y: torch.tensor, sigma: float = 1.0) -> torch.tensor:
        """
        正解ラベルを正規分布に変換する処理

        Args:
            y (torch.tensor(batch_size)): 正規分布の平均
            sigma (float): 正規分布の分散

        Returns:
            torch.tensor(batch_size, self.dist_size): 正規分布
        """
        batch_size = y.size(0)
        X = torch.arange(0, self.dist_size).repeat(batch_size, 1).to(self.device)
        N = torch.exp(-torch.square(X - y.unsqueeze(1)) / (2 * sigma**2))
        d = torch.sqrt(torch.tensor(2 * math.pi * sigma**2)).to(self.device)
        
        return N / d