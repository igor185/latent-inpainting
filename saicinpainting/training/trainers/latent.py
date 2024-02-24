import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss, masked_l2_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class LatentTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(
            **rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

        # freeze encoder and decoder
        # for params in self.generator.model[:5].parameters():
        #     params.requires_grad = False
        #
        # for params in self.generator.model[-13:].parameters():
        #     params.requires_grad = False

        # path = "/home/engineer/Dev/igor/thesis/own/output_auto_lama_4(another_mask_encoder)/model.pth"
        # state = torch.load(path, map_location='cpu')
        # state_dict = {i[16:] if i.startswith('generator.') else i: state[i] for i in state}
        # self.generator.model.load_state_dict(state_dict, strict=True)


    def forward(self, batch):
        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        # load auto-encoder
        img = batch['image']
        mask = batch['mask']
        # encoder = self.generator.model[:5]
        # lama_inner = self.generator.model[5:-13]
        # decoder = self.generator.model[-13:]
        # mask_encoder = self.mask_encoder

        masked_img = img * (1 - mask)  # (1 - mask) * self.config.mask.fill_value

        # if self.add_noise_kwargs is not None:
        #     noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
        #     if self.noise_fill_hole:
        #         masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
        #     masked_img = torch.cat([masked_img, noise], dim=1)
        #
        # if self.concat_mask:
        #     masked_img = torch.cat([masked_img, mask], dim=1)

        # batch["h_gt"] = encoder(img)
        # batch["h"] = encoder(masked_img)
        # batch["h_mask"] = mask_encoder(mask)
        # batch["ft_h_mask"] = (batch["h"][0] + batch["h_mask"][:, :batch["h"][0].shape[1]],
        #                       batch["h"][1] + batch["h_mask"][:, batch["h"][0].shape[1]:])

        batch["feat_gt"] = self.autoencoder.encoder(img)
        batch["feat_masked"] = self.autoencoder.encoder(masked_img)

        batch["mask_feat"] = self.mask_encoder(mask)
        batch["feat_merged"] = self.re_embeder(batch["feat_masked"], batch["mask_feat"])

        batch["refined_feat"] = self.unet(batch["feat_merged"])
        batch["predicted_feat"] = batch["refined_feat"] + batch["feat_masked"]

        batch['predicted_image'] = self.autoencoder.decoder(batch["predicted_feat"])
        batch['gt_image'] = self.autoencoder.decoder(batch["feat_gt"])

        batch["refined_feat_decoded"] = self.autoencoder.decoder(batch["refined_feat"])
        batch["expected_refined_feat_decoded"] = self.autoencoder.decoder(batch["feat_gt"]-batch["feat_masked"])

        # # Merge using Roman's method
        # img_feat = torch.cat(batch["h"], dim=1)
        # mask_feat = batch["h_mask"]
        #
        # img_feat_skip = self.conv1(img_feat)
        # merge = self.conv4(torch.cat([self.conv3(img_feat), mask_feat], dim=1))
        # batch["ft_h_mask"] = img_feat_skip + self.conv5(self.conv2(img_feat_skip) + merge)
        # batch["ft_h_mask"] = (batch["ft_h_mask"][:, :batch["h"][0].shape[1]], batch["ft_h_mask"][:, batch["h"][0].shape[1]:])


        # batch['refined_h'] = lama_inner(batch["ft_h_mask"])
        # lamb = self.generator.lamb
        # batch['refined_h'] = (lamb*batch['refined_h'][0] + batch["h"][0], lamb*batch['refined_h'][1] + batch["h"][1])

        # batch['predicted_image'] = decoder(batch["refined_h"])

        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        predicted_feat = batch['predicted_feat']
        feat_gt = batch['feat_gt']
        img = batch['gt_image']
        predicted_img = batch["predicted_image"]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        l2_value = torch.nn.functional.mse_loss(predicted_feat, feat_gt)

        # L1
        # l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
        #                           self.config.losses.l1.weight_known,
        #                           self.config.losses.l1.weight_missing)

        l1_in_mask = masked_l1_loss(predicted_img, img, original_mask, 0, 1)
        l2_in_mask = masked_l2_loss(predicted_img, img, original_mask, 0, 1)

        total_loss = l2_value #+ l2_in_mask
        metrics = dict(gen_l2=l2_value, l1_in_mask=l1_in_mask, l2_in_mask=l2_in_mask) #lamb=self.generator.lamb)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img,
                                    mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        # mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        # self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
        #                                          generator=self.generator, discriminator=self.discriminator)
        # discr_real_pred, discr_real_features = self.discriminator(img)
        # discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        # adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
        #                                                                  fake_batch=predicted_img,
        #                                                                  discr_real_pred=discr_real_pred,
        #                                                                  discr_fake_pred=discr_fake_pred,
        #                                                                  mask=mask_for_discr)
        # total_loss = total_loss + adv_gen_loss
        # metrics['gen_adv'] = adv_gen_loss
        # metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        # if self.config.losses.feature_matching.weight > 0:
        #     need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
        #     mask_for_fm = supervised_mask if need_mask_in_fm else None
        #     fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
        #                                      mask=mask_for_fm) * self.config.losses.feature_matching.weight
        #     total_loss = total_loss + fm_value
        #     metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics
