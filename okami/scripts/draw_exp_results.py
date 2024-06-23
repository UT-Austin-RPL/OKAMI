import os

import init_path
import argparse
import json
import pprint
import time

import cv2
import numpy as np
import h5py
import shutil

from robot.gr1 import GR1URDFModel
from utils.video_utils import VideoWriter
import deoxys_vision.utils.transformation.transform_utils as T

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str)
    return parser.parse_args()

def add_text_on_image(img, text, pos=(10,30), color=(255,255,255), fontsize=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontsize
        fontColor = color
        lineType = 2
        cv2.putText(img, text, 
            pos, 
            font, 
            fontScale,
            fontColor,
            lineType)
        return img

def main():
    args = parse_args()

    folder = os.path.join("data", args.task_name)

    dirs = os.listdir(folder)
    dirs = sorted(dirs)

    res = []
    for d in dirs:
        if not os.path.isdir(os.path.join(folder, d)):
            continue
        with open(os.path.join(folder, d, "result.json"), 'r') as file:
            exp_res = json.load(file)
        img = cv2.imread(os.path.join(folder, d, "initial_frame.png"))
        res.append((img, exp_res))

    # display all images and text information (success, failure, reason) in one image
    n = len(res)

    print("Number of results: ", n, "how do you want to display them?")
    r = eval(input("Please input number of rows:"))
    c = eval(input("Please input number of columns:"))

    h = 480
    w = 640
    img = np.zeros((h*r, w*c, 3), dtype=np.uint8)
    success_num = 0
    for i in range(n):
        new_img = cv2.resize(res[i][0], (w, h))
        # cv2.putText(new_img, res[i][1]['success'], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        text_str = "Success" if res[i][1]['success'] else f"Failure: {res[i][1]['reason']}"
        new_img = add_text_on_image(new_img, text_str, (100, 250), color=(200, 0, 100), fontsize=1.3)

        if res[i][1]['success']:
            success_num += 1

        img[i//c*h:(i//c+1)*h, i%c*w:(i%c+1)*w, :] = new_img

        # img[i*h:(i+1)*h, :, :] = cv2.resize(res[i][0], (w, h))
        # cv2.putText(img, res[i][1]['success'], (0, i*h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, res[i][1]['reason'], (0, i*h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    new_img = np.zeros((h, w, 3), dtype=np.uint8)
    new_img = add_text_on_image(new_img, f"success rate: {success_num / n}", (100, 250), color=(200, 0, 100), fontsize=1.3)
    img[n//c*h:(n//c+1)*h, n%c*w:(n%c+1)*w, :] = new_img

    cv2.imwrite(os.path.join(folder, "task_results.png"), img)

if __name__ == '__main__':
    main()    