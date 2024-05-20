import logging
import os
import sys
import traceback

import matplotlib.pyplot as plt

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

IGNORE_MASK = False
CONCAT = False
index = 2
name = f'/home/engineer/Dev/igor/thesis/own/output/tmp2/{"" if not IGNORE_MASK else "no_mask_"}result_{index}'
os.environ['freeze'] = ''
@hydra.main(config_path='../configs/prediction', config_name='latent_control.yaml')
def main(predict_config: OmegaConf):
    # Change process working directory to the project root
    # os.chdir(hydra.utils.get_original_cwd())

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
    # train_config.generator.ngf = 8
    # checkpoint_path = "/mnt/code/logs/lama/experiments/engineer_2023-11-03_20-45-09_train_lama-autoencoder.yaml_/models/last.ckpt"

    # checkpoint_path = ("/mnt/code/logs/lama/experiments/engineer_2024-05-09_22-18-47_train_lama-fourier-latent-l2_lama-fourier-l2/models/last.ckpt")

    # checkpoint_path = ("/mnt/code/logs/lama/experiments/engineer_2024-05-08_23-14-26_train_lama-fourier-latent-l2_lama-fourier-mask_encoder_pixel_l2_control/models/last.ckpt")
    model, _ = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)


    # sd = SDWrapper()
    # sd.model.to(device)

    images, masks = get_dataset()
    if not predict_config.indir.endswith('/'):
        predict_config.indir += '/'
    dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

    encoder = model.generator.model[:5].eval()
    refiner = model.generator.model[5:-16]
    decoder = model.generator.model[-13:].eval()

    s = sum([i.mean() for i in list(encoder.parameters())]).cpu().float()
    sd = sum([i.mean() for i in list(decoder.parameters())]).cpu().float()
    if not torch.allclose(s, torch.tensor(5.3772)) and not torch.allclose(sd, torch.tensor(2.8756)):
        print(s, sd)
    # assert torch.allclose(s, torch.tensor(5.3772)), f"Sum of the mean of the parameters of the encoder is {s}"
    # assert torch.allclose(sd,
    #                       torch.tensor(2.875640392303467)), f"Sum of the mean of the parameters of the decoder is {sd}"

    mask_encoder = model.mask_encoder

    def forward(batch):
        mask = batch["mask"].float()
        masked_img = batch['image'] * (1 - batch['mask'])
        if not CONCAT:
            ms = []
            m = mask
            for l in mask_encoder:
                m = l(m)
                ms.append(m)

        else:
            masked_img = torch.cat([masked_img, mask], dim=1)
        #     (decoder(encoder(masked_img)) - masked_img).abs().mean()

        a, b = encoder(batch['image']), encoder(masked_img)
         # = (a[0] - b[0], a[1] - b[1])
        for i, l in enumerate(encoder):
            masked_img = l(masked_img)
            if not IGNORE_MASK:
                if i == 1:
                    masked_img = model.zero_conv2(ms[2]) + masked_img[0], masked_img[1]
                if i == 2:
                    masked_img = model.zero_conv3(ms[5]) + masked_img[0], masked_img[1]
                if i == 3:
                    masked_img = model.zero_conv4(ms[8]) + masked_img[0], masked_img[1]
                if i == 4:
                    out = model.zero_conv5(ms[11])
                    masked_img = out[:, :masked_img[0].shape[1]] + masked_img[0], out[:, masked_img[0].shape[1]:] + \
                                 masked_img[1]
        masked_feat = masked_img


        delta_e = (a[0] - masked_feat[0], a[1] - masked_feat[1])

        batch["delta_exp"] = decoder(delta_e)
        batch["masked_feat"] = masked_feat
        batch["masked_feat_decode"] = decoder(masked_feat)

        refined_feat = refiner(masked_feat)
        batch["refined_feat"] = refined_feat
        delta = (refined_feat[0] - masked_feat[0], refined_feat[1] - masked_feat[1])
        batch["delta_decoded"] = decoder(delta)
        pred_img = decoder(refined_feat)

        batch['predicted_image'] = pred_img
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        return batch


    with torch.no_grad():
        batch_lama = default_collate([dataset[index]])
        batch_lama = move_to_device(batch_lama, device)
        batch_lama['mask'] = (batch_lama['mask'] > 0) * 1
        # res = model(batch_lama)
        batch_lama = forward(batch_lama)

        cv2.imwrite(f'{name}_inpainted_raw.png', to_img(batch_lama['predicted_image']))
        cv2.imwrite(f'{name}delta_exp.png', to_img(batch_lama["delta_exp"]))
        # cv2.imwrite(f'{name}_inpainted.png', to_img(batch_lama['inpainted']))
        zd = batch_lama['masked_feat_decode']
        cv2.imwrite(f'{name}_before_ref.png', to_img(batch_lama['masked_feat_decode']))
        delta = batch_lama['delta_decoded']
        im = to_img(delta)
        imr = cv2.resize(im, (256, 256))
        cv2.imwrite(f'{name}_delta.png', to_img(delta))

def to_img(tensor):
    res = tensor * 255
    res = res[0].cpu().numpy()
    res = np.transpose(res, (1, 2, 0))
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res.astype(np.uint8)



if __name__ == '__main__':
    main()