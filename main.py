import lib
import yaml
import os, sys
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    opts = lib.get_args()

    loaders = lib.get_loaders(opts)

    validation_batch = next(iter(loaders['valid']))
    model = lib.Model(opts, validation_batch=validation_batch)

    if opts['wandb']:
        wandb.init(
            project='IQA', 
            name=opts['name'],
            config=opts
        )
        logger = WandbLogger()
    else:
        logger = None

    MyModelCheckpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{opts['name']}/",
        filename=f'date={lib.today()}_' + '{val_srocc:.3f}_{epoch}',
        **opts['model_checkoint'])

    MyEarlyStopping = EarlyStopping(**opts['early_stopping'])

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        accelerator='gpu',
        devices=[opts['device']],
        callbacks=[MyEarlyStopping, MyModelCheckpoint],
        log_every_n_steps=1,
    )

    trainer.fit(model, loaders['train'], loaders['valid'])

    model.test_dashboard = 'test_koniq'
    trainer.test(model, loaders['train_koniq'])

    model.test_dashboard = 'test_clive'
    trainer.test(model, loaders['test_clive'])

if __name__ == '__main__':
    main()