from omegaconf import OmegaConf
import torch
import sys
import glob
import os
import tqdm
from torch import nn

sys.path.append("sd")
from sd.scripts.inpaint import make_batch
from sd.ldm.util import instantiate_from_config

# input_dir = "/home/engineer/Dev/igor/thesis/sd/data/inpainting_examples"
input_dir = "/home/engineer/Dev/igor/thesis/own/data"

class SDWrapper(nn.Module):
    def __init__(self):
        super(SDWrapper, self).__init__()
        config = OmegaConf.load("/home/engineer/Dev/igor/thesis/sd/models/ldm/inpainting_big/config.yaml")
        self.model = instantiate_from_config(config.model)
        state_dict = torch.load("/home/engineer/Dev/igor/thesis/sd/models/ldm/inpainting_big/last.ckpt", map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

    def encoder(self, image):
        return self.model.cond_stage_model.encode(image, return_intermediate=False)

    def decoder(self, features):
        return self.model.decode_first_stage(features, False)

    def encoder_features(self, inp):
        co, c = self.model.cond_stage_model.encode(inp, return_intermediate=True)
        return co, c

    def encoder_inner_features(self, co):
        c = self.model.cond_stage_model.encoder.conv_out(co)
        c = self.model.cond_stage_model.quant_conv(c)
        return self.decoder_features(c)


    def encoder_image(self, inp):
        c = self.model.cond_stage_model.encode(inp, return_intermediate=False)
        return c

    def decoder_features(self, features):
        return self.model.decode_first_stage(features, False)

    def to_np_image(self, features):
        predicted_image = torch.clamp((features + 1.0) / 2.0,
                                      min=0.0, max=1.0)
        image = predicted_image[0].permute(1, 2, 0).cpu().numpy() * 255
        image = image.astype("uint8")
        return image

def get_dataset():
    masks = sorted(glob.glob(os.path.join(input_dir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")
    return images, masks

def main():

    model = SDWrapper()
    images, masks = get_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")


    model.model.to(device)
    with torch.no_grad():
        with model.model.ema_scope():
            image, mask = images[0], masks[0]
            batch = make_batch(image, mask, device=device)

            co, c = model.encoder_features(batch["image"])

            decoded = model.decoder_features(c)
            image = model.to_np_image(decoded)
            # c_image = model.to_np_image(c)


if __name__ == '__main__':
    main()