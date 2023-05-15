import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

class TRAINER():
    def __init__(self, model, device, train_loder, val_loder, criterion, optimizer, scheduler):
        self.model = model.to(device)
        self.device = device
        self.train_loder = train_loder
        self.val_loder = val_loder
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.softmax = nn.Softmax().to(device)

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        train_loss = 0
        for i, (src, tgt, ans) in enumerate(self.train_loder):
            print(f'batch={i+1}/{len(self.train_loder)}', end='\r')
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            ans = ans.to(self.device)

            out = self.model(src, tgt)
            loss = self.criterion(out, ans)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()
        
        return train_loss / (i+1)

    def validation(self):
        self.model.eval()
        val_loss = 0
        logits = []
        self.labels= []
        with torch.no_grad():
            for i, (src, tgt, ans) in enumerate(self.val_loder):
                print(f'batch={i+1}/{len(self.val_loder)}', end='\r')
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                ans = ans.to(self.device)

                out = self.model(src, tgt)
                loss = self.criterion(out, ans)
                val_loss += loss.item()

                # # 記録
                # out = self.softmax(out)
                # logit = out.cpu().numpy().tolist()
                # label = ans.item()
                # logits.append(logit[0])
                # self.labels.append(label)
        self.logits = np.array(logits)

        return val_loss / (i+1)

    def chaek_performance(self):

        auc = roc_auc_score(self.labels, self.logits[:, 1])
        pred = []
        for log in self.logits:
            p = np.argmax(log)
            pred.append(p)

        cm = confusion_matrix(self.labels, pred)
        tn, fp, fn, tp = cm.flatten()

        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return auc, accuracy, precision, recall, f1