import logging
import os
import sys
import traceback

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import nn, optim

# sys.path.append('sd')

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_train_dataloader

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


LOGGER = logging.getLogger(__name__)

from omegaconf import OmegaConf
import torch
import yaml
import hydra
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
import pytorch_lightning as L
import torch.nn.functional as F


class AutoEncoderModel(nn.Module):
    def __init__(self, lama):
        super().__init__()
        self.lama = lama
        self.encoder = lama.generator.model[:5]
        self.decoder = lama.generator.model[-13:]

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()
        self.config = config


    def forward(self, x):
        return self.model(x)

    def _get_reconstruction_loss(self, batch):
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)


@hydra.main(config_path='../configs/training', config_name='lama-autoencoder.yaml')
def main(predict_config: OmegaConf):
    register_debug_signal_handlers()

    # train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    # with open(train_config_path, 'r') as f:
    #     train_config = OmegaConf.create(yaml.safe_load(f))
    #
    # train_config.training_model.predict_only = True
    # train_config.visualizer.kind = 'noop'
    #
    # checkpoint_path = os.path.join(predict_config.model.path,
    #                                'models',
    #                                predict_config.model.checkpoint)
    # lama = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')

    for params in lama.parameters():
        params.requires_grad = True

    dataloader = make_default_train_dataloader(**train_config.data.train)
    model = Autoencoder(AutoEncoderModel(lama), train_config)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )






if __name__ == '__main__':
    main()