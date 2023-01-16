import lib
from argparse import ArgumentParser
import yaml
import os, sys
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    parser = ArgumentParser()
    parser.add_argument('--saliency', type=int, required=True, help='use saliency or not [0 | 1]')
    parser.add_argument('--name', type=str, required=True, help='name of experiment')
    parser.add_argument('--device', type=int, required=True, help='index of cuda device')
    args = parser.parse_args()
    opts = vars(args)
    opts['saliency'] = bool(opts['saliency'])
    opts.update(lib.get_default_opts())

    loader_train, loader_valid, loader_test_koniq, loader_test_clive = \
        lib.get_loaders(opts)

    model = lib.Model(opts, validation_batch=next(iter(loader_valid)))

    wandb.init(
        project='IQA', 
        name=opts['name'],
        config=opts
    )

    wandb_logger = WandbLogger()

    MyModelCheckpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{opts['name']}/",
        filename=f'date={lib.today()}_' + '{val_srocc:.3f}_{epoch}',
        **opts['model_checkoint'])

    MyEarlyStopping = EarlyStopping(**opts['early_stopping'])

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=100,
        accelerator='gpu',
        devices=[opts['device']],
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        log_every_n_steps=1,
    )

    trainer.fit(model, loader_train, loader_valid)

    model._test_dashboard = 'test_koniq'
    trainer.test(model, loader_test_koniq)

    model._test_dashboard = 'test_clive'
    trainer.test(model, loader_test_clive)

if __name__ == '__main__':
    main()