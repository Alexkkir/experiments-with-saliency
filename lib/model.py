import torch
import numpy as np
from torch import nn
import torchvision
from scipy import stats
import pytorch_lightning as pl

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, activation=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if activation:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )
        else:
            self.layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
            )

    def forward(self, x):
        return self.layers(x)

class Model(pl.LightningModule):
    def __init__(self, saliency_flg, alpha_sal=0.2):
        super().__init__()

        backbone = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        backbone = list(backbone.children())[:-2]
        backbone = nn.Sequential(*backbone)
        self.backbone = backbone
        
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LinearBlock(1408, 1024, 0.25),
            LinearBlock(1024, 256, 0.25),
            LinearBlock(256, 1, 0, activation=False)
        )

        self.sal_conv = nn.Conv2d(1408, 1, (1, 1), 1, 0)
        self.mse_loss = nn.MSELoss()
        self.alpha_sal = alpha_sal if saliency_flg is True else 0
        self.saliency_flg = saliency_flg
        self._test_dashboard = 'test'

    def saliency_loss(self, pred, y):
        pred = pred / pred.mean()
        y = y / y.mean()
        return ((pred - y) ** 2).mean()

    def forward(self, x):
        x = self.backbone(x)
        if self.saliency_flg:
            saliency = self.sal_conv(x)
            x = saliency * x # fusion
        x = self.mlp(x)

        if self.saliency_flg:
            return x, saliency
        else:
            return x

    def training_step(self, batch, batch_idx):
        return self._step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def _step(self, batch):
        x, sal_target, y = batch['image'], batch['saliency'], batch['subj_mean']

        if self.saliency_flg:
            pred, sal_pred = self(x)
        else:
            pred = self(x)
        pred = pred.flatten()

        loss = self.mse/loss(pred, y) * (1 - self.alpha_sal)
        if self.saliency_flg:
            loss += self.saliency/loss(sal_pred, sal_target) * self.alpha_sal

        true = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return {'loss': loss, 'results': (true, pred)}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 3e-5},
            {'params': self.mlp.parameters(), 'lr': 3e-4}
        ], weight_decay=3e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.2, 
            patience=5, 
            verbose=True)
            
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/srocc"
        } 

        return [optimizer], [lr_dict]

    def training_epoch_end(self, outputs):
        """log and display average test loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.print(f"| TRAIN plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {loss:.2f}" )

        self.log('train/loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train/plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train/srocc', srocc, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.print(f"[Epoch {self.trainer.current_epoch:3}] VALID plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {loss:.2f}", end= " ")

        self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val/plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val/srocc', srocc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_srocc', srocc, prog_bar=False, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):
        """log and display average test loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.log(f'{self._test_dashboard}/loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{self._test_dashboard}/plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{self._test_dashboard}/srocc', srocc, prog_bar=True, on_epoch=True, on_step=False)

    def _epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        true = np.concatenate([x['results'][0] for x in outputs])
        predicted = np.concatenate([x['results'][1] for x in outputs])
        
        plcc = stats.pearsonr(predicted, true)[0]
        srocc = stats.spearmanr(predicted, true)[0]
        return loss, plcc, srocc