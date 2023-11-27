import logging
import os
import sys
import traceback

from torch import nn

sys.path.append('.')

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
import matplotlib.pyplot as plt
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

from omegaconf import OmegaConf
import torch
import yaml
import hydra
from saicinpainting.training.trainers import load_checkpoint
from own.vae_inference import get_dataset, SDWrapper
from saicinpainting.utils import register_debug_signal_handlers
from sd.scripts.inpaint import make_batch


@hydra.main(config_path='../configs/prediction', config_name='autoencoder.yaml')
def main(predict_config: OmegaConf):
    register_debug_signal_handlers()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0")
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    train_config.generator.ngf = 8
    train_config.generator.input_nc = 3
    #
    # checkpoint_path = os.path.join(predict_config.model.path,
    #                                'models',
    #                                predict_config.model.checkpoint)
    # checkpoint_path = "/mnt/code/logs/lama/experiments/engineer_2023-11-06_09-09-55_train_lama-autoencoder.yaml_/last.ckpt"
    # checkpoint_path = "/mnt/code/logs/lama/experiments/engineer_2023-11-03_20-45-09_train_lama-autoencoder.yaml_/models/last.ckpt"
    checkpoint_path = "/home/engineer/Dev/igor/thesis/own/output_auto_lama_4(another_mask_encoder)/model.pth"
    mask_encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
    ).to(device)
    lama, mask_encoder = load_checkpoint(train_config, checkpoint_path, mask_encoder, strict=False, map_location='cpu')
    # model.freeze()
    for params in lama.parameters():
        params.requires_grad = True

    lama.to(device)

    encoder = lama.generator.model[:5]
    lama_inner = lama.generator.model[5:-13]
    decoder = lama.generator.model[-13:]

    for params in encoder.parameters():
        params.requires_grad = False

    for params in decoder.parameters():
        params.requires_grad = False



    images, masks = get_dataset()

    optimizer = torch.optim.Adam([*lama.parameters(), *mask_encoder.parameters()], lr=0.0001)
    steps = 50000
    log_freq = 200
    batch_size = 4
    mask_type = "net"  # net or zero or no
    name = "output_auto_lama_4(another_mask_encoder)"
    predict_only = True

    with torch.no_grad():
        input, mask = images[0], masks[0]
        input1, mask1 = images[1], masks[1]
        batch = make_batch(input, mask, device=device)
        batch1 = make_batch(input1, mask1, device=device)
        # mask_down = F.interpolate(batch["mask"], scale_factor=8, mode="nearest")
        # mask_embeddings = mask_encoder(mask_down)
        feat = encoder(batch["image"])
        # feat = feat + mask_embeddings
        # feat = lama_inner(feat)
        decoded = decoder(feat)
        image_gt = (decoded.permute(0, 2, 3, 1)[0].cpu().numpy() * 255).astype("uint8")
        cv2.imwrite("/home/engineer/Dev/igor/thesis/own/" + name + "/image_gt.png",
                    cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    losses = []
    mask_loss = []
    img_losses = []
    mask_down = batch["mask"]  # F.interpolate(batch["mask"], scale_factor=0.125, mode="nearest")
    for i in tqdm.tqdm(range(steps)):
        if mask_type == "net":
            mask_embeddings = mask_encoder(mask_down)
        elif mask_type == "zero":
            mask_embeddings = mask_encoder(torch.zeros_like(mask_down))
        elif mask_type == "ones":
            mask_embeddings = mask_encoder(torch.ones_like(mask_down))
        elif mask_type == "no":
            mask_embeddings = None
        else:
            mask_embeddings = None

        # batch_input = torch.cat([batch["masked_image"], batch["mask"]], dim=1)
        feat_masked = encoder(batch["masked_image"])
        if mask_embeddings is not None:
            amount = mask_embeddings.shape[1] // 4
            mask_embeddings1, mask_embeddings2 = mask_embeddings[:, :amount], mask_embeddings[:, amount:]
            # feat_masked = (feat_masked[0] + mask_embeddings1, feat_masked[1] + mask_embeddings2)
            feat_masked = (mask_embeddings1, mask_embeddings2)

        feat_masked_l = lama_inner(feat_masked)
        feat_masked_l = (feat_masked_l[0] + feat_masked[0], feat_masked_l[1] + feat_masked[1])
        image_pred = decoder(feat_masked_l)

        feat_masked_l_cat = torch.cat(feat_masked_l, dim=1)
        feat_cat = torch.cat(feat, dim=1)
        if not predict_only:
            loss = torch.nn.functional.mse_loss(feat_masked_l_cat,
                                                feat_cat)  # + torch.nn.functional.mse_loss(image_pred, batch["image"])
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % log_freq == 0:
                with torch.no_grad():
                    # loss_mask = np.mean(np.square(image[gt_mask] - batch["image"][gt_mask]))
                    image_pred = image_pred.permute(0, 2, 3, 1)[0].cpu().numpy()
                    image_gt_b = batch["image"].permute(0, 2, 3, 1)[0].cpu().numpy()
                    img_loss = np.mean(np.square(image_pred - image_gt_b))
                    m = batch["mask"].permute(0, 2, 3, 1)[0].cpu().numpy() > 0
                    m = np.tile(m, (1, 1, 3))
                    in_mask_loss = np.mean(np.square(image_pred[m] - image_gt_b[m]))
                    print(f"Loss: {loss.item()}, Image loss: {img_loss}")  # , Mask loss: {0},

                    img_losses.append(img_loss)
                    mask_loss.append(in_mask_loss)
                    image_pred = (image_pred * 255).astype("uint8")
                    cv2.imwrite(f"/home/engineer/Dev/igor/thesis/own/" + name + f"/{i:09d}.png",
                                cv2.cvtColor(image_pred, cv2.COLOR_RGB2BGR))
                    plt.plot(losses)
                    plt.savefig("/home/engineer/Dev/igor/thesis/own/" + name + "/losses.png")
                    plt.close()
                    #
                    plt.plot(mask_loss)
                    plt.savefig("/home/engineer/Dev/igor/thesis/own/" + name + "/mask_loss.png")
                    plt.close()

                    plt.plot(img_losses)
                    plt.savefig("/home/engineer/Dev/igor/thesis/own/" + name + "/img_loss.png")
                    plt.close()
        else:
            with torch.no_grad():
                image_pred = image_pred.permute(0, 2, 3, 1)[0].cpu().numpy()
                image_pred = (image_pred * 255).astype("uint8")
                cv2.imwrite(f"/home/engineer/Dev/igor/thesis/own/" + name + f"/predict.png",
                            cv2.cvtColor(image_pred, cv2.COLOR_RGB2BGR))
                break
    # Save the model
    torch.save(lama.state_dict(), "/home/engineer/Dev/igor/thesis/own/" + name + "/model.pth")
    torch.save(mask_encoder.state_dict(), "/home/engineer/Dev/igor/thesis/own/" + name + "/mask_encoder.pth")


if __name__ == '__main__':
    main()
