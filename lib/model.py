import torch
import numpy as np
from torch import nn
import torchvision
from scipy import stats
import pytorch_lightning as pl
from .constants import MEAN, STD
import wandb
from torchvision.transforms.functional import resize


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
    def __init__(self, opts, validation_batch=None):
        super().__init__()

        self.opts = opts

        backbone = torchvision.models.efficientnet_b2(
            weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
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

        N_CHANNELS_IN_LAST_LAYER = 1408
        self.sal_conv = nn.Conv2d(N_CHANNELS_IN_LAST_LAYER, 1, (1, 1), 1, 0)
        self.mse_loss = nn.MSELoss()
        self.test_dashboard = 'test'
        self.validation_batch = validation_batch

    def saliency_loss(self, pred, y):
        pred = pred / pred.mean()
        y = y / y.mean()
        return ((pred - y) ** 2).mean()

    def forward(self, x):
        x = self.backbone(x)
        saliency = self.sal_conv(x)
        x = saliency * x  # fusion
        x = self.mlp(x)

        return x, saliency

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def _step(self, batch):
        x, sal_target, y = batch['image'], batch['saliency'], batch['subj_mean']

        pred, sal_pred = self(x)

        pred = pred.flatten()

        loss = self.mse_loss(pred, y) * (1 - self.alpha_sal) + self.saliency_loss(sal_pred, sal_target) * self.alpha_sal

        true = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return {'loss': loss, 'results': (true, pred)}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 3e-5},
            {'params': self.mlp.parameters(), 'lr': 3e-4}
        ], weight_decay=3e-4)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **self.opts['lr_scheduler']
        )

        lr_dict = {
            "scheduler": lr_scheduler,
            **self.opts['lr_dict']
        }

        return [optimizer], [lr_dict]

    def training_epoch_end(self, outputs):
        """log and display average test loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.print(
            f"| TRAIN plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {loss:.2f}")

        self.log('train/srocc', srocc, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('train/plcc', plcc, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('train/loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.print(
            f"[Epoch {self.trainer.current_epoch:3}] VALID plcc: {plcc:.2f}, srocc: {srocc:.2f}, loss: {loss:.2f}", end=" ")

        self.log('val/srocc', srocc, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val/plcc', plcc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_srocc', srocc, prog_bar=False,
                 on_epoch=True, on_step=False)

        # log saliency maps
        if self.opts['visualize_saliency']:
            N = 8
            images = self.validation_batch['image'][:N]
            shape = images.shape[-2:]

            saliency = self.validation_batch['saliency'][:N]
            saliency = torch.concat([saliency] * 3, dim=1)

            _, saliency_pred = self(images.to(self.device))
            saliency_pred = saliency_pred.detach().cpu()
            saliency_pred = torch.concat([saliency_pred] * 3, dim=1)

            saliency = resize(saliency, shape)
            saliency_pred = resize(saliency_pred, shape)

            images = images.permute(0, 2, 3, 1) * STD + MEAN
            images = images.permute(0, 3, 1, 2)

            for i in range(N):
                saliency_pred[i] = (saliency_pred[i] - saliency_pred[i].min()) / \
                    (saliency_pred[i].max() - saliency_pred[i].min())

            grid = torchvision.utils.make_grid(torch.concat(
                [saliency, images, saliency_pred])).permute(1, 2, 0)
            grid = wandb.Image(grid.numpy())
            self.logger.log_image(
                key='grid with sal maps',
                images=[grid]
            )

    def test_epoch_end(self, outputs):
        """log and display average test loss and accuracy"""
        loss, plcc, srocc = self._epoch_end(outputs)

        self.log(f'{self.test_dashboard}/loss', loss,
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{self.test_dashboard}/plcc', plcc,
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{self.test_dashboard}/srocc', srocc,
                 prog_bar=True, on_epoch=True, on_step=False)

    def _epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        true = np.concatenate([x['results'][0] for x in outputs])
        predicted = np.concatenate([x['results'][1] for x in outputs])

        plcc = stats.pearsonr(predicted, true)[0]
        srocc = stats.spearmanr(predicted, true)[0]
        return loss, plcc, srocc
