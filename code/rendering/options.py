import os
import logging
from collections import namedtuple
import argparse
import json
import json5
import numpy as np
from easydict import EasyDict
import torch
from distributed import (
    get_rank,
    synchronize
)
class TrainOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--nump", type=int, default=1)
        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        
        return
        

    def parse_args(self, suffix=''):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        device = "cuda"
        n_gpu = torch.cuda.device_count()
        print('n_gpu:', n_gpu)
        self.distributed = self.args.nump >= 1
        #self.distributed = False
        
        if self.distributed:
            local_rank=int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=self.args.nump)
            synchronize()
        else:
            local_rank = self.args.local_rank
        print('local_rank:', self.args.local_rank)
        name = self.args.name
        #
        
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            from_json = self.args.from_json
            logging.warning('A config file is passed, overriding all arguments with the values from the config file!')
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json5.load(f)
                # json_args['log_dir'] = os.path.join(os.path.abspath(json_args['log_dir']), json_args['name'])
                # json_args['summary_dir'] = os.path.join(os.path.abspath(json_args['log_dir']), json_args['name'])
                # json_args = namedtuple("json_args", json_args.keys())(**json_args)

            # self.args = json_args
            self.args = EasyDict(**json_args)
            self.args.from_json = from_json

        self.args.name = name
        self.args.local_rank = local_rank
        self.args.distributed = self.distributed
        print('distributed:', self.distributed)
        self.save_dump()

        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard'+suffix)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints'+suffix)
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        #self.args.local_rank = self.local_rank
        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config_videoavatar.json.
        """
        exp_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return

class TestOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--nump", type=int, default=1)
        self.parser.add_argument("--train_name", type=str, default='')
        self.parser.add_argument("--test_name", type=str, default='')
        self.parser.add_argument('--smooth', type=bool, default=True)
        self.parser.add_argument('--no_head_no_hand', type=bool, default=True)
        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        
        return
        

    def parse_args(self, suffix=''):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        n_gpu = torch.cuda.device_count()
        print('n_gpu:', n_gpu)
        self.distributed = self.args.nump >= 1
        
        if self.distributed:
            local_rank=int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=self.args.nump)
            synchronize()
        else:
            local_rank = self.args.local_rank
        print('local_rank:', self.args.local_rank)
        name = self.args.name
        #
        
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            from_json = self.args.from_json
            logging.warning('A config file is passed, overriding all arguments with the values from the config file!')
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json5.load(f)
            self.args = EasyDict(**json_args)
            self.args.from_json = from_json

        self.args.name = name
        self.args.local_rank = local_rank
        self.args.distributed = self.distributed
        print('distributed:', self.distributed)
        self.save_dump()

        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard'+suffix)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints'+suffix)
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config_videoavatar.json.
        """
        exp_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
