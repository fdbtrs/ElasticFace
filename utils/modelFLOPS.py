import logging

from pytorch_model_summary import summary

import torch

from utils.countFLOPS import count_model_flops

from backbones.iresnet import iresnet100


from config.config import config as cfg

if __name__ == "__main__":
    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size)
    elif cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size)
    else:
        backbone = None
        logging.info("load backbone failed!") 

    print(summary(backbone, torch.zeros((1, 3, 112, 112)), show_input=False))

    flops = count_model_flops(backbone)
    
    print(flops)

    #model.eval()
    #tic = time.time()

    #model.forward(torch.zeros((1, 3, 112, 112)))
    #end = time.time()
    #print(end-tic)
