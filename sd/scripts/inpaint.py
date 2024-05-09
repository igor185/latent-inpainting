import argparse, os, sys, glob
sys.path.append(".")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
  # to run from the root folder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def make_batch(image, mask, device=None):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        if device is not None:
            batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

# Make dataloader
class DataLoader:
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __getitem__(self, item):
        image = self.images[item]
        mask = self.masks[item]
        batch = make_batch(image, mask)
        batch["path"] = image
        return batch

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        default="data/inpainting_examples",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="data/inpainting_examples_out",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()
    inp_dir = opt.indir

    # folders = ["random_medium_512",  "random_thick_256",  "random_thick_512"]
    # folders = ["random_thin_256",  "random_thin_512",  "seg_512"]
    folders = ["random_medium_256"]

    for folder in folders:
        print(f"Processing {folder}...")

        opt.indir = f"{inp_dir}/{folder}"
        opt.outdir = f"data/inpainting_examples_out/{folder}"

        masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask000.png")))
        images = [x.replace("_mask000.png", ".png") for x in masks]
        print(f"Found {len(masks)} inputs.")
        dataset = DataLoader(images, masks)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=False)

        config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                              strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        # with torch.no_grad():
        with model.ema_scope():
            for batch in tqdm(data_loader):
                batch["image"], batch["mask"], batch["masked_image"] = batch["image"].to(device), batch["mask"].to(device), batch["masked_image"].to(device)

                pathes = batch["path"]

                # batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)*255
                for i, path in enumerate(pathes):
                    out_path = os.path.join(opt.outdir, os.path.split(path)[1])
                    Image.fromarray(inpainted[i].astype(np.uint8)).save(out_path)
