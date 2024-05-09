import logging
import torch

from saicinpainting.training.trainers.autoencoder import AutoencoderTrainingModule
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
from saicinpainting.training.trainers.latent import LatentTrainingModule


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule
    elif kind == 'autoencoder':
        return AutoencoderTrainingModule
    elif kind == 'latent':
        return LatentTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, mask_encoder=None, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    state = torch.load(path, map_location=map_location)
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
        state_mask = torch.load("/home/engineer/Dev/igor/thesis/own/output_auto_lama_4(another_mask_encoder)/mask_encoder.pth", map_location=map_location)
        mask_encoder.load_state_dict(state_mask, strict=strict)
    model.on_load_checkpoint(state)
    return model, mask_encoder
