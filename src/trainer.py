import torch

class TRAINER():
    def __init__(self, model, device, train_loder, val_loder, criterion, optimizer, scheduler):
        self.model = model.to(device)
        self.device = device
        self.train_loder = train_loder
        self.val_loder = val_loder
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        train_loss = 0
        for i, (src, tgt, ans) in enumerate(self.train_loder):
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
        self.logits = []
        self.labels= []
        with torch.no_grad():
            for i, (src, tgt, ans) in enumerate(self.val_loder):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                ans = ans.to(self.device)

                out = self.model(src, tgt)
                loss = self.criterion(out, ans)
                val_loss += loss.item()

        return val_loss / (i+1)