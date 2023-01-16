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
    parser.add_argument('--saliency', type=bool, required=True, help='use saliency or not')
    parser.add_argument('--name', type=str, required=True, help='name of experiment')
    args = parser.parse_args()
    opts = vars(args)
    opts_yaml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
    opts.update(opts_yaml)

    loader_train, loader_valid, loader_test_koniq, loader_test_clive = \
        lib.get_loaders(opts)

    model = lib.Model(use_saliency=opts['saliency'])

    wandb.init(
        project='IQA', 
        name=opts['name'],
        config=opts
    )

    wandb_logger = WandbLogger()

    MyModelCheckpoint = ModelCheckpoint(dirpath='checkpoints/',
                                    filename=f'date={lib.today()}_' + '{val_srocc:.3f}_{epoch}',
                                    monitor='val_srocc', 
                                    mode='max', 
                                    save_top_k=1,
                                    save_weights_only=True,
                                    verbose=False)

    MyEarlyStopping = EarlyStopping(monitor = "val_srocc", 
                                    mode = "max",
                                    patience = 15,
                                    verbose = True)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=100,
        accelerator='gpu',
        devices=[0],
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