import argparse
import logging
import os

import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
import torch
from torchvision import transforms

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import training.backbones


from  sklearn.metrics.pairwise import cosine_similarity

from training.utils.utils_callbacks import CallBackVerification
from training.utils.utils_logging import init_logging

sys.path.append('/root/xy/work_dir/xyface/')
from torch.nn.parallel import DistributedDataParallel
from training.config import config as cfg



def eval(args):
    eval_one_epoch = False
    iter_eval = 295672
    step_size = 11372
    output_folder = args.output_folder
    model_folder = args.model_folder

    log_root = logging.getLogger()
    init_logging(log_root, 0, output_folder, logfile="test.log")
    callback_verification = CallBackVerification(step_size, 0, cfg.val_targets, cfg.rec)
    dropout = 0.4 if cfg.dataset == "webface" else 0

    if eval_one_epoch:
        backbone = eval("backbones.{}".format(cfg.net_name))(False, dropout=dropout, fp16=cfg.fp16)
        model_path = os.path.join(model_folder, str(iter_eval) + "backbone.pth")
        backbone.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(backbone, device_ids=[args.gpu_id])
        print(f"LOADED MODEL FROM {model_path}")
        callback_verification(iter_eval, model)
    else:
        weights = os.listdir(model_folder)

        for w in weights:
            if "backbone" in w:
                backbone = eval("backbones.{}".format(cfg.net_name))(False, dropout=dropout, fp16=cfg.fp16)

                backbone.load_state_dict(torch.load(os.path.join(model_folder, w)))
                model = torch.nn.DataParallel(backbone, device_ids=[args.gpu_id])
                callback_verification(int(w.split("backbone")[0]), model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--gpu_id', type=int, default=0, help='local_rank')
    parser.add_argument('--model_folder', type=str, default="output/emore_random_resnet", help='local_rank')
    parser.add_argument('--output_folder', type=str, default="output/emore_random_resnet", help='local_rank')

    args_ = parser.parse_args()
    eval(args_)
