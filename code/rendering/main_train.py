import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from options import TrainOptions
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5,6"
n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('n_gpu0:', torch.cuda.device_count())
def modify_opt_according_to_os(opt):
    import platform
    if platform.system() == 'Windows':
        opt.num_workers = 0     # pytorch multi-threading doesn't work well on windowns

    return opt


if __name__ == '__main__':
    opt = TrainOptions().parse_args()
    
    

    trainer_module = __import__(opt.trainer_module, fromlist=['Trainer'])
    trainer = trainer_module.Trainer(opt)
    trainer.train()
    logging.info('Training Done. ')
    trainer.test_seq(start_frame_id=100, end_frame_id=7500, cam_ids=[0])

