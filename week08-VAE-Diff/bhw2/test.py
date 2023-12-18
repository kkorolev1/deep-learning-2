import logging
import json
import os
from pathlib import Path
import sys

import torch
from tqdm import tqdm, trange
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from piq import FID, ssim
from torchvision.utils import save_image

from hw_diff.utils import ROOT_PATH
from hw_diff.model.DDPM.diffusion import Diffusion
from hw_diff.model.DDPM.utils import get_named_beta_schedule
from hw_diff.collate_fn import collate

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

def normalize_01(tensor):
    return (tensor + 1) / 2

def metric_collate(batch):
    return {
        "images": torch.cat([normalize_01(item.unsqueeze(0)) for item in batch])
    }

def get_dataloader(config: DictConfig, collate_fn):
    bs = config["batch_size"]
    dataset = instantiate(config["dataset"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )
    return dataloader

def get_tensor_dataloader(config: DictConfig, tensor):
    bs = config["batch_size"]
    dataset = TensorDataset(tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, collate_fn=metric_collate,
        shuffle=False,
        drop_last=False
    )
    return dataloader

@hydra.main(version_base=None, config_path="hw_diff/configs", config_name="test")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    logger = logging.getLogger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = instantiate(config["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    logger.info(f"Device {device}")
    model = model.to(device)
    model.eval()

    beta_schedule = config["beta_schedule"]
    T = config["T"]
    diffusion = Diffusion(betas=get_named_beta_schedule(beta_schedule, T))
    
    batch_size = config["batch_size"]
    eval_shape = config["eval_shape"]
    metric_dataloader = get_dataloader(config, metric_collate)
    num_iters = len(metric_dataloader.dataset) // batch_size
    gen_images = None
    for _ in trange(num_iters, desc="Inference the model"):
        gen_batch_images = diffusion.p_sample_loop(model, (batch_size, *eval_shape), device)
        gen_batch_images = torch.clip(gen_batch_images, min=-1, max=1)
        if gen_images is None:
            gen_images = gen_batch_images
        else:
            gen_images = torch.cat((gen_images, gen_batch_images), dim=0)
    
    gen_images = 0.5 + gen_images / 2
    
    fid_metric = FID().to(device)
    gt_feats = fid_metric.compute_feats(metric_dataloader)
    
    gen_dataloader = get_tensor_dataloader(config, gen_images)
    gen_feats = fid_metric.compute_feats(gen_dataloader)

    fid = fid_metric(gt_feats, gen_feats)
    logger.info(f"FID {fid}")
    
    gt_dataloader = get_dataloader(config, collate)
    gt_images = None
    for batch in tqdm(gt_dataloader):
        if gt_images is None:
            gt_images = batch["image"]
        else:
            gt_images = torch.cat((gt_images, batch["image"]), dim=0)
    ssim_metric = ssim(gt_images.cpu(), gen_images.cpu(), data_range=1.)
    logger.info(f"SSIM {ssim_metric}")
    
    save_image(gen_images, config["output_path"])

if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
