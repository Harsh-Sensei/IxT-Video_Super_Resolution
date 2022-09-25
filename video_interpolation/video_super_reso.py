import argparse
import cv2
import numpy as np
import os

import numpy as np
import math
import cv2
import importlib
import json
import torch
from collections import OrderedDict
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
sys.path.append("./")
H_LIM = 256 
W_LIM = 256


from image_super_resolution.CARN_pytorch.carn.infer import infer

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

cfg = {
    "group": 1,
    "ckpt_path": "./image_super_resolution/CARN_pytorch/checkpoint/carn.pth",
    "model": "carn",
    "scale": 4,
    "shave": 20,
}
cfg = Dict2Class(cfg)

module = importlib.import_module("image_super_resolution.CARN_pytorch.carn.model.{}".format(cfg.model))

def image_super_resolution(img):

    net = module.Net(multi_scale=True,
                     group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    device = torch.device("cpu")
    net = net.to(device)

    lr = Image.fromarray(np.uint8(img))
    lr = lr.convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return infer(net, device, transform(lr), cfg, return_img=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Super Resolution")
    parser.add_argument('--test_video', type=str, default=None)
    parser.add_argument('--test_image', type=str, default=None)
    parser.add_argument('--stack_bicubic', type=bool, default=False)
    args = parser.parse_args()

    if args.test_image is not None:
        frame = cv2.imread(args.test_image)
        height, width, _ = frame.shape
        if height > H_LIM or width > W_LIM:
            raise Exception("Frame shape exceed limits")
        sr_frame = image_super_resolution(frame)*255
        if args.stack_bicubic:
            bi_frame = cv2.resize(frame, (frame.shape[1]*cfg.scale, frame.shape[0]*cfg.scale), cv2.INTER_CUBIC).astype('float32')
            sr_frame = np.hstack([bi_frame, sr_frame])
        cv2.imwrite("sr_test_image.png", sr_frame)

    if args.test_video is not None:
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(args.test_video)
        height, width = int(cap.get(4)), int(cap.get(3))
        size_f = (width, height)
        result = cv2.VideoWriter("sr_test_video.mp4", 
                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                        20, size_f)

        if height > H_LIM or width > W_LIM:
            raise Exception("Frame shape exceed limits")

        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening test video file")
        if (result.isOpened()==False):
            print("Error opening save video file")

        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == False:
                raise Exception("Video reading error")

            sr_frame = image_super_resolution(frame)*255
            if args.stack_bicubic:
                bi_frame = cv2.resize(frame, (frame.shape[1]*cfg.scale, frame.shape[0]*cfg.scale), cv2.INTER_CUBIC).astype('float32')
                sr_frame = np.hstack([bi_frame, sr_frame])
            sr_frame = sr_frame.astype('uint8')
            result.write(sr_frame)

        # When everything done, release
        # the video capture object
        cap.release()
        result.release()

        # Closes all the frames
        cv2.destroyAllWindows()

