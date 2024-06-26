import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
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

        # set seed
        torch.manual_seed(42)
        self.x = torch.rand(1, 3, 256, 256)

    def forward(self, batch, mode='train'):
        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        use_latent_l2 = self.config.generator.get("use_latent_l2", False)
        use_control_guidance = self.config.generator.get("use_control_guidance", False)

        encoder = self.generator.model[:5]
        refiner = self.generator.model[5:-13]
        decoder = self.generator.model[-13:]
        # assert if the sum of the mean of the parameters of the encoder is equal to 5.3772

        # x= self.x.to(encoder[1].ffc.convl2l.weight.device)
        # for l in encoder:
        #     x = l(x)
        #     print(x.mean() if type(x) == torch.Tensor else x[0].mean())
        # tensor(0.5007, device='cuda:0')
        # tensor(0.6494, device='cuda:0')
        # tensor(0.3892, device='cuda:0')
        # tensor(0.5692, device='cuda:0')
        # tensor(0.4528, device='cuda:0')
        # print("----")
        # for l in decoder:
        #     x = l(x)
        #     print(x.mean() if type(x) == torch.Tensor else x[0].mean())

        # tensor(0.5060, device='cuda:0')
        # tensor(-0.1422, device='cuda:0')
        # tensor(0.0211, device='cuda:0')
        # tensor(0.4433, device='cuda:0')
        # tensor(-0.0154, device='cuda:0')
        # tensor(0.0498, device='cuda:0')
        # tensor(0.4458, device='cuda:0')
        # tensor(0.0759, device='cuda:0')
        # tensor(0.0143, device='cuda:0')
        # tensor(0.3823, device='cuda:0')
        # tensor(0.3802, device='cuda:0')
        # tensor(-0.2725, device='cuda:0')
        # tensor(0.4388, device='cuda:0')
        # s = sum([i.mean() for i in list(encoder.parameters())]).cpu().float()
        # sd = sum([i.mean() for i in list(decoder.parameters())]).cpu().float()
        # if not torch.allclose(s, torch.tensor(5.3772)) and not torch.allclose(sd, torch.tensor(2.8756)):
        #     print(s, sd)
        # assert torch.allclose(s, torch.tensor(5.3772)), f"Sum of the mean of the parameters of the encoder is {s}"
        # assert torch.allclose(sd, torch.tensor(2.875640392303467)), f"Sum of the mean of the parameters of the decoder is {sd}"
        #
        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask and not self.config.generator.get("use_mask_encoder", False):
            masked_img = torch.cat([masked_img, mask], dim=1)

        if encoder[1].ffc.convl2l.in_channels == 4:
            masked_feat = encoder(masked_img)
        elif use_control_guidance:
            ms = []
            m = mask.float()
            for l in self.mask_encoder:
                m = l(m)
                ms.append(m)

            for i, l in enumerate(encoder):
                masked_img = l(masked_img)
                if i == 1:
                    masked_img = self.zero_conv2(ms[2]) + masked_img[0], masked_img[1]
                if i == 2:
                    masked_img = self.zero_conv3(ms[5]) + masked_img[0], masked_img[1]
                if i == 3:
                    masked_img = self.zero_conv4(ms[8]) + masked_img[0], masked_img[1]
                if i == 4:
                    out = self.zero_conv5(ms[11])
                    masked_img = out[:, :masked_img[0].shape[1]] + masked_img[0], out[:, masked_img[0].shape[1]:] + masked_img[1]
                # xs.append(masked_img)
            masked_feat = masked_img

        elif self.config.generator.get("use_mask_encoder", False):
            masked_feat = encoder(masked_img)
            mask_feat = self.mask_encoder(mask.float())
            masked_feat = (masked_feat[0] + mask_feat[:, :16], masked_feat[1] + mask_feat[:, 16:])
        else:
            masked_feat = encoder[2:](self.generator.mask_conv(encoder[0](masked_img)))
        refined_feat = refiner(masked_feat)
        # refined_feat = (masked_feat[0] + refined_feat[0],  masked_feat[1] + refined_feat[1])
        batch["refined_feat"] = refined_feat
        pred_img = decoder(refined_feat)
        if use_latent_l2:
            encoder.eval()
            batch['gt_feat'] = encoder(img)
            # batch['gt_feat'] = (a.detach(), b.detach())
            # batch["img_auto"] = decoder(batch['gt_feat'])
            # ((img - batch["img_auto"]) ** 2).mean()
        batch['predicted_image'] = pred_img
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

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
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        if 'gt_feat' in batch:
            l2_latent = 0.0001 * F.l1_loss(torch.cat(batch['refined_feat'], dim=1), torch.cat(batch['gt_feat'], dim=1))
            total_loss = total_loss + l2_latent
            metrics['gen_l2_latent'] = l2_latent

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

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
