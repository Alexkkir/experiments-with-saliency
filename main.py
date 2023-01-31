import lib
import yaml
import os, sys
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    opts = lib.get_args()

    loaders = lib.get_loaders(opts, fast=True)
    validation_batch = next(iter(loaders['valid']))

    model = lib.Model(opts, validation_batch=validation_batch)

    if opts['wandb']:
        wandb.init(
            entity='alexkkir',
            project='IQA', 
            name=opts['name'],
            config=opts
        )
        logger = WandbLogger()
    else:
        logger = None

    CheckpointLast = ModelCheckpoint(
        dirpath=f"checkpoints/{opts['name']}/",
        filename=f'last_date={lib.today()}_' + '{val_srocc:.3f}_{epoch}',
        **opts['model_checkpoint_last'])

    CheckpointBest = ModelCheckpoint(
        dirpath=f"checkpoints/{opts['name']}/",
        filename=f'best_date={lib.today()}_' + '{val_srocc:.3f}_{epoch}',
        **opts['model_checkpoint_best'])

    MyEarlyStopping = EarlyStopping(**opts['early_stopping'])

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=opts['max_epochs'],
        accelerator='gpu',
        devices=[opts['device']],
        callbacks=[MyEarlyStopping, CheckpointLast, CheckpointBest],
        log_every_n_steps=1,
    )

    trainer.fit(model, loaders['train'], loaders['valid'])
    print(CheckpointBest.)
    print(CheckpointLast.best_model_path)

    model.test_dashboard = 'test_koniq_best'
    trainer.test(ckpt_path='best', dataloaders=loaders['test_koniq'])

    model.test_dashboard = 'test_koniq_last'
    trainer.test(ckpt_path='last', dataloaders=loaders['test_koniq'])

    model.test_dashboard = 'test_clive_best'
    trainer.test(ckpt_path='best', dataloaders=loaders['test_clive'])

    model.test_dashboard = 'test_clive_last'
    trainer.test(ckpt_path='last', dataloaders=loaders['test_clive'])

if __name__ == '__main__':
    main()