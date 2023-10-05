import logging
import os
import sys
import traceback
# sys.path.append('sd')

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

from omegaconf import OmegaConf
import torch
import yaml
import hydra
from saicinpainting.training.trainers import load_checkpoint
from own.vae_inference import get_dataset, SDWrapper
from saicinpainting.utils import register_debug_signal_handlers
from sd.scripts.inpaint import make_batch

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    register_debug_signal_handlers()

    device = torch.device(predict_config.device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path,
                                   'models',
                                   predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    sd = SDWrapper()
    sd.model.to(device)

    images, masks = get_dataset()
    if not predict_config.indir.endswith('/'):
        predict_config.indir += '/'
    dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

    with torch.no_grad():
        image, mask = images[0], masks[0]
        batch = make_batch(image, mask, device=device, source="sd")
        batch_lama = default_collate([dataset[0]])
        batch_lama = move_to_device(batch_lama, device)
        batch_lama['mask'] = (batch_lama['mask'] > 0) * 1
        batch["concat"] = torch.cat([batch["masked_image"], batch["mask"]], dim=1)
        co, c = sd.encoder_features(batch["image"])

        co_1, co_2 = co[:, :128], co[:, 128:]
        co_1, co_2 = model.generator.model[5:-14]((co_1, co_2))
        co = torch.cat([co_1, co_2], dim=1)

        decoded = sd.decoder_features(co)
        result = model.generator(batch["concat"])



if __name__ == '__main__':
    main()