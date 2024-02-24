#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define the layers
        self.conv1 = conv(3, 64)
        self.block1 = Block(64, 64)
        self.conv2 = conv(64, 64, stride=2, bias=False)
        self.block2 = Block(64, 64)
        self.block3 = Block(64, 64)
        self.block4 = Block(64, 64)
        self.conv3 = conv(64, 64, stride=2, bias=False)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 64)
        self.conv4 = conv(64, 64, stride=2, bias=False)
        self.block8 = Block(64, 64)
        self.block9 = Block(64, 64)
        self.block10 = Block(64, 64)
        self.final_conv = conv(64, 4)

    def forward(self, x):
        # Forward pass through the layers
        x = self.conv1(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv3(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.conv4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.final_conv(x)
        return x

    def load_pretrained_weights(self, pretrained_dict):
        # Create a new state dictionary with remapped keys
        new_state_dict = {}

        # Define a mapping from the sequential model's keys to the encoder's keys
        key_mapping = {
            '0': 'conv1',
            '1': 'block1',
            '2': 'conv2',
            '3': 'block2',
            '4': 'block3',
            '5': 'block4',
            '6': 'conv3',
            '7': 'block5',
            '8': 'block6',
            '9': 'block7',
            '10': 'conv4',
            '11': 'block8',
            '12': 'block9',
            '13': 'block10',
            '14': 'final_conv'
        }

        # Remap keys
        for key in pretrained_dict.keys():
            # Split the key into layer index and parameter
            split_key = key.split('.')
            layer_idx = split_key[0]
            param = '.'.join(split_key[1:])  # Join the remaining parts back if there are any

            if layer_idx in key_mapping:
                # Construct the new key and add it to the new state dictionary
                new_key = f"{key_mapping[layer_idx]}.{param}"
                new_state_dict[new_key] = pretrained_dict[key]

        # Load the new state dictionary into the encoder
        self.load_state_dict(new_state_dict, strict=False)


def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="/home/engineer/Dev/igor/thesis/weights/taesd_encoder.pth", decoder_path="/home/engineer/Dev/igor/thesis/weights/taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_pretrained_weights(torch.load(encoder_path, map_location="cpu"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)
