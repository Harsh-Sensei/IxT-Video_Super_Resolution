import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def infer(net, device, image, cfg, return_img=False):
    lr = image

    t1 = time.time()
    lr = lr.unsqueeze(0).to(device)
    sr = net(lr, cfg.scale).detach().squeeze(0)
    lr = lr.squeeze(0)
    t2 = time.time()

    if return_img:
        return sr.permute([1, 2, 0]).numpy()
    
    model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
    sr_dir = os.path.join(cfg.sample_dir,
                            model_name,
                            "x{}".format(cfg.scale),
                            "SR")
    
    bicubic_img = torchvision.transforms.Resize((image.shape*cfg.scale),
                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    os.makedirs(sr_dir, exist_ok=True)

    sr_im_path = os.path.join(sr_dir, "{}".format("sample_SR.jpg"))
    bicubic_path = os.path.join(sr_dir, "{}".format("sample_bicubic.jpg"))
    
    save_image(sr, sr_im_path)
    save_image(bicubic_img, bicubic_path)
    print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
        .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))


def main(cfg):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            module = importlib.import_module("model.{}".format(cfg.model))
            net = module.Net(multi_scale=True,
                             group=cfg.group)
            print(json.dumps(vars(cfg), indent=4, sort_keys=True))

            state_dict = torch.load(cfg.ckpt_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

            device = torch.device("cpu")
            net = net.to(device)

            lr = Image.open("./carn/sample.jpg")
            lr = lr.convert("RGB")

            transform = transforms.Compose([
                    transforms.ToTensor()
                ])

            infer(net, device, transform(lr), cfg)
 
    print(prof.key_averages().table(sort_by="gpu_memory_usage", row_limit=10))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
