import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from options import TestOptions


def modify_opt_according_to_os(opt):
    import platform
    if platform.system() == 'Windows':
        opt.num_workers = 0     # pytorch multi-threading doesn't work well on windowns

    return opt


if __name__ == '__main__':

    opt = TestOptions().parse_args()

    trainer_module = __import__(opt.trainer_module, fromlist=['Trainer'])
    trainer = trainer_module.Trainer(opt)
    trainer.test_seq(cam_ids=[0], start_frame_id=0)



    logging.info('Training Done. ')

