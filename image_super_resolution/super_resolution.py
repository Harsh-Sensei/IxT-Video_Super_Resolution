import argparse
import cv2 as cv
import numpy as np
import torch

def rock_segment(image):
    pass

def classification(image):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical flow')
    parser.add_argument('--webcam', type=bool, default=False)

    args = parser.parse_args()

    if args.webcam:
        capture = cv.VideoCapture(0)
        while True:
            ret_val, frame = capture.read()
            