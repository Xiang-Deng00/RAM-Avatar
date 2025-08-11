import logging
import os, glob
import warnings
import copy
import functools

import cv2 as cv
import numpy as np
import trimesh
from datetime import datetime
import json
import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# custom modules
from lib.utils.base_trainer import BaseTrainer
from lib.utils.data_loader import CheckpointDataLoader
from lib.networks.triplane import Triplane
from lib.networks.styleunet.styleunet import SWGAN_unet, Discriminator, TriPlane_Conv
import lib.ops.losses as loss_op

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


class Trainer(BaseTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

    def init_fn(self):
        self.vgg = VGGLoss('cuda')
        opt = self.options
        self.train_dir_rgb = os.path.join(self.options.log_dir, 'sample')
        if not os.path.exists(self.train_dir_rgb):
            os.mkdir(self.train_dir_rgb)
        print('train_dir_rgb:', self.train_dir_rgb)
        dataset_module = __import__(opt.dataset_module, fromlist=['Dataset'])
        self.train_ds = dataset_module.Dataset(opt.dataset_dir, opt.cam_num, opt.cam_ids_to_use,
                                               opt.pretrained_smpl_param_ckpt_path,
                                               first_frame=opt.first_frame, last_frame=opt.last_frame,
                                               white_bkgd=opt.white_bkgd, training=True, encode_smpl_map=False, 
                                               resolution=opt.img_res, config=opt
                                               )
        if hasattr(opt, 'dataset_module_test'):
            dataset_module_test = __import__(opt.dataset_module_test, fromlist=['Dataset'])
            self.test_ds = dataset_module_test.Dataset(opt.dataset_dir, opt.cam_num, opt.cam_ids_to_use,
                                                       opt.pretrained_smpl_param_ckpt_path,
                                                       first_frame=opt.first_frame, last_frame=opt.last_frame,
                                                       white_bkgd=opt.white_bkgd, training=False, encode_smpl_map=False, 
                                               resolution=opt.img_res, config=opt)
        else:
            self.test_ds = self.train_ds
        logging.info('Found %d training data items' % len(self.train_ds))

        self._create_net(opt)
        self._create_loss_and_optimizer(opt)

        self.models_dict = {
            'triplane': self.triplane,
            'diffnet': self.diffnet,
            'triplane_conv': self.triplane_conv,
            'discriminator_t': self.discriminator_t,
            'stylecode': self.style_code,
        }
        self.optimizers_dict = {
            'optm_diffnet': self.optm_diffnet,
            'optm_discriminator': self.optm_discriminator,
        }
        if self.options.pretrained_checkpoint is not None:
            logging.info('Loading pre-trained checkpoint from %s... ' % self.options.pretrained_checkpoint)
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        logging.info('#trainable_params = %d' % (
            sum(p.numel() for p in self.diffnet.parameters() if p.requires_grad)
        ))

    def _create_net(self, opt):
        self.style_code_dim = 512
        self.triplane = Triplane(16, opt.triplane_res).to(self.device)
        self.diffnet = SWGAN_unet(
            inp_size=self.train_ds.resolution//1,
            inp_ch=(16*3 + 16 * 2) ,
            out_ch=3,
            out_size=self.train_ds.resolution,
            style_dim=self.style_code_dim,
            n_mlp=2).to(self.device)
        self.triplane_conv = TriPlane_Conv(48, 96, 48).to(self.device)

        

        self.discriminator_t = Discriminator(
            self.train_ds.resolution,
            img_channel=9).to(self.device)
        
        print('style_length:', self.options.last_frame+1)
        self.style_code = nn.Embedding(self.options.last_frame+1, 512, max_norm=1.0).to(self.device)
        nn.init.normal_(self.style_code.weight.data, 0, 1/np.sqrt(512))

        self.multi_gpu = False
        if self.multi_gpu:
            self.triplane = torch.nn.DataParallel(self.triplane)
            self.diffnet = torch.nn.DataParallel(self.diffnet)
            self.triplane_conv = torch.nn.DataParallel(self.triplane_conv)
            self.discriminator_t = torch.nn.DataParallel(self.discriminator_t)

        if opt.distributed:
            self.triplane = nn.parallel.DistributedDataParallel(self.triplane, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False, find_unused_parameters=False)
            self.diffnet = nn.parallel.DistributedDataParallel(self.diffnet, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False, find_unused_parameters=False)
            self.discriminator_t = nn.parallel.DistributedDataParallel(self.discriminator_t, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False, find_unused_parameters=False)
            self.style_code = nn.parallel.DistributedDataParallel(self.style_code, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False, find_unused_parameters=False)
            self.triplane_conv = nn.parallel.DistributedDataParallel(self.triplane_conv, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False, find_unused_parameters=False)
            self.local_rank = opt.local_rank
        self.distributed = opt.distributed


    def _create_loss_and_optimizer(self, opt):
        # losses
        logging.debug('Constructing losses... ')
        self.color_l1_loss = nn.SmoothL1Loss(beta=0.05).to(self.device)
        self.color_mse_loss = nn.MSELoss().to(self.device)
        self.hand_bbox_loss = loss_op.HandPerceptualLoss(reduction='mean').to(self.device)

        # optimizer
        logging.debug('Constructing optimizer... ')
        self.optm_diffnet = torch.optim.Adam(params=list(self.diffnet.parameters())+list(self.triplane.parameters())+
                                                    list(self.style_code.parameters()),#+list(self.triplane_conv.parameters()),
                                             lr=float(self.options.lr))
        self.optm_discriminator = torch.optim.Adam(params=list(self.discriminator_t.parameters()),#+list(self.discriminator_s.parameters()),
                                                   lr=float(self.options.lr))

    def set_train(self):
        for k in self.models_dict.keys():
            self.models_dict[k].train()

    def set_eval(self):
        for k in self.models_dict.keys():
            self.models_dict[k].eval()

    def get_cond(self, input_batch):
        input_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in input_batch.items()}
        smpl_map = input_batch['smpl_map']
        cond = self.triplane(smpl_map.permute(0, 2, 3, 1))

        
        return cond
    
    def get_cond2(self, input_batch):
        input_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in input_batch.items()}
        smpl_map = input_batch['smpl_map2']
        cond = self.triplane(smpl_map.permute(0, 2, 3, 1))
        return cond
    

    def get_loss_gen(self, input_batch, cond, gan_loss_weight, hand_loss_weight, vgg_loss_w, input_batch_list, test=False):
        
        idx = input_batch['idx']
        if not test:
            img_gt = input_batch['img']
            hand_bbox = input_batch['hand_bbox']
        if 'hand_skel_map' in input_batch:
            cond = torch.cat([cond, self.diffnet.forward(styles=[self.style_code(idx)], condition_img=input_batch['hand_skel_map'], forward_hand=True)], dim=1)
        if 'face' in input_batch:
            cond = torch.cat([cond, self.diffnet.forward(styles=[self.style_code(idx)], condition_img=input_batch['face'], forward_face=True)], dim=1)
        if test:
            idx = torch.zeros(1).cuda().long()
            self.style_code_no = self.style_mean
        else:
            self.style_code_no = self.style_code(idx)
        img_pred = self.diffnet(
            [self.style_code_no],
            condition_img=cond,
            cond=None,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=False,
        )
        if test:
            return 0, img_pred
        img_gt_list_pred = []
        for i in range(len(input_batch_list)):
            if i == (len(input_batch_list) // 2):
                img_gt_list_pred.append(img_pred)
            else:
                img_gt_list_pred.append(input_batch_list[i]['img'])
        fake_pred_t = self.discriminator_t(torch.cat(img_gt_list_pred, dim=1))
        l1_loss = self.color_l1_loss(img_pred, img_gt)
        loss_gan_s = loss_op.g_nonsaturating_loss(fake_pred_t) * gan_loss_weight
        loss_hand = self.hand_bbox_loss(img_pred, img_gt, hand_bbox)* hand_loss_weight
        vgg_loss = self.vgg(img_pred, img_gt) * vgg_loss_w

        diff_loss = l1_loss * 5 + loss_gan_s + loss_hand + vgg_loss
        return diff_loss, img_pred
    
    def get_loss_dis(self, input_batch, img_pred, gan_loss_weight, d_reg_every):
 
        
        img_gt_list = []
        img_gt_list_pred = []
        for i in range(len(input_batch)):
            img_gt_list.append(input_batch[i]['img'])
            if i == (len(input_batch) // 2):
                img_gt_list_pred.append(img_pred.detach())
            else:
                img_gt_list_pred.append(input_batch[i]['img'])
        fake_pred = self.discriminator_t(torch.cat(img_gt_list, dim=1))
        real_pred = self.discriminator_t(torch.cat(img_gt_list_pred, dim=1))
        loss_dis = loss_op.d_logistic_loss(real_pred, fake_pred) 
        loss_dis = loss_dis * gan_loss_weight

        return loss_dis
 
    def train_step(self, input_batch):
        gan_weight = 1.1 ** (self.step_count // 2000)
        gan_loss_weight = min(1e-3 * gan_weight, 0.02)
        hand_loss_weight = 0.1
        d_reg_every = 16
        vgg_loss_w = 0.03
        self.set_train()
        require = True
        if require:
            requires_grad(self.triplane, True)
            requires_grad(self.diffnet, True)
            requires_grad(self.style_code, True)
            requires_grad(self.triplane_conv, True)
            requires_grad(self.discriminator_t, False)
        # cond = self.get_cond(input_batch[1])
        cond = self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond(input_batch[0])) + self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond2(input_batch[1]))
               
        

        diff_loss, img_pred = self.get_loss_gen(input_batch[1], cond, gan_loss_weight=gan_loss_weight, hand_loss_weight=hand_loss_weight, vgg_loss_w=vgg_loss_w, input_batch_list=input_batch)

        self.optm_diffnet.zero_grad()

        
        diff_loss.backward()
        nn.utils.clip_grad_norm_(self.triplane.parameters(), max_norm=1, norm_type=2)
        nn.utils.clip_grad_norm_(self.diffnet.parameters(), max_norm=1, norm_type=2)
        nn.utils.clip_grad_norm_(self.style_code.parameters(), max_norm=1, norm_type=2)
        self.optm_diffnet.step()



        if require:
            requires_grad(self.triplane, False)
            requires_grad(self.diffnet, False)
            requires_grad(self.style_code, False)
            requires_grad(self.triplane_conv, False)
            requires_grad(self.discriminator_t, True)
        
        loss_dis = self.get_loss_dis(input_batch, img_pred, gan_loss_weight=gan_loss_weight, d_reg_every=d_reg_every)
        self.optm_discriminator.zero_grad()
        loss_dis.backward()
        nn.utils.clip_grad_norm_(self.discriminator_t.parameters(), max_norm=1, norm_type=2)

        self.optm_discriminator.step()

        

        losses = {
            'dif': diff_loss.item(),
            'dis': loss_dis.item(),
        }

        if self.step_count % 200 == 0 or self.step_count < 20:
            self.log_sample(losses, input_batch, img_pred)

        return losses
    
    def log_sample(self, losses, input_batch, img_pred):
        if self.distributed:
            if self.local_rank != 0:
                return

        self.print_loss(losses)
        img_gt = input_batch[0]['img']
        smpl_map = input_batch[0]['smpl_map']
        sample = img_pred[0].permute(1, 2, 0).detach().cpu().numpy()
        sample = np.uint8(np.clip(sample*127.5+127.5, 0, 255))
        img_gt = img_gt[0].permute(1, 2, 0).detach().cpu().numpy()
        img_gt = np.uint8(np.clip(img_gt*127.5+127.5, 0, 255))

        smpl_map = smpl_map[0].permute(1, 2, 0).detach().cpu().numpy()
        smpl_map[:, :, 1] -= 0.4
        smpl_map = (smpl_map * 0.3 + 0.5) * 255.0

        skel_map = input_batch[0]['hand_skel_map'][0].permute(1, 2, 0).detach().cpu().numpy()
        
        skel_map = skel_map * 255.0
        skel_map[...,2] += skel_map[...,3]
        skel_map[...,2] += skel_map[...,4]
        skel_map = skel_map[...,:3]

        face = input_batch[0]['face'][0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
        smpl_map = cv.resize(smpl_map, (sample.shape[0], sample.shape[1]), interpolation=cv.INTER_NEAREST)
        
        skel_map = cv.resize(skel_map, (sample.shape[0], sample.shape[1]), interpolation=cv.INTER_NEAREST)
        sample = np.concatenate([img_gt, sample, smpl_map, skel_map, face], axis=1)
        #print('sample:', sample.shape)
        path_write = os.path.join(self.train_dir_rgb, str(self.step_count).zfill(8) + '.png')
        cv.imwrite(path_write, sample[:, :, ::-1])

    def zero_grad(self):
        for k, optim in self.optimizers_dict.items():
            optim.zero_grad()

    def step(self):
        for k, net in self.models_dict.items():
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.)
        for k, optim in self.optimizers_dict.items():
            optim.step()

    def zerograd_backward_update(self, total_loss):
        self.zero_grad()
        total_loss.backward()
        self.step()

    def update_lr(self):
        # update learning rate
        if self.step_count < self.options.warm_up_end_step:
            learning_factor = self.step_count / self.options.warm_up_end_step
        else:
            alpha = 0.05
            progress = (self.step_count - self.options.warm_up_end_step) / (self.options.end_step - self.options.warm_up_end_step)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        learning_rate = self.options.lr * learning_factor
        for k, optim in self.optimizers_dict.items():
            if k == 'optm_diffnet':
                for param_group in optim.param_groups:
                    param_group['lr'] = learning_rate
            else:
                for param_group in optim.param_groups:
                    param_group['lr'] = learning_rate #* 0.1
        return learning_rate

    def clip_norm(self, tensor, max_norm=None):
        tensor_norm = torch.norm(tensor, dim=-1, keepdim=True)
        if max_norm is not None:
            tensor_norm = torch.maximum(tensor_norm, torch.ones_like(tensor_norm) * max_norm)
            tensor = tensor / (tensor_norm + 1e-10)
        return tensor, tensor_norm

    def print_loss(self, losses, skip_keys=()):
        str = ''
        for k, v in losses.items():
            if k not in skip_keys and k.find('total_loss')==-1:
                str += '%s: %.3e, ' % (k, (float(v.item()) if isinstance(v, torch.Tensor) else float(v)))

        for k, v in losses.items():
            if k not in skip_keys and k.find('total_loss')!=-1:
                str += '%s: %.3e, ' % (k, (float(v.item()) if isinstance(v, torch.Tensor) else float(v)))

        logging.info(str)

    def train_summaries(self, input_batch, losses=None):
        assert losses is not None
        for ln in losses.keys():
            self.summary_writer.add_scalar(ln, losses[ln], self.step_count)

    def gen(self, input_batch, cond, idx):
        if 'hand_skel_map' in input_batch:
            cond = torch.cat([cond, self.diffnet.forward(styles=[self.style_code(idx)], condition_img=input_batch['hand_skel_map'], forward_hand=True)], dim=1)
        if 'face' in input_batch:
            cond = torch.cat([cond, self.diffnet.forward(styles=[self.style_code(idx)], condition_img=input_batch['face'], forward_face=True)], dim=1)
        img_pred = self.diffnet(
                        [self.style_code_no],
                        condition_img=cond,
                        cond=None,
                        return_latents=False,
                        inject_index=None,
                        truncation=1,
                        truncation_latent=None,
                        input_is_latent=False,
                        noise=None,
                        randomize_noise=False,
                    )
        return img_pred

    def test_seq(self, start_frame_id=None, end_frame_id=None, resume=False, cam_ids=None):
        out_dir_rgb = os.path.join(self.options.log_dir, 'step_%d' % self.step_count, 'reconstruction_rgb')
        os.makedirs(out_dir_rgb, exist_ok=True)
        self.set_eval()
        self.triplane = self.triplane.half()
        self.triplane_conv = self.triplane_conv.half()
        self.diffnet = self.diffnet.half()
        start_frame_id = 0 if start_frame_id is None else start_frame_id
        end_frame_id = self.test_ds.frame_num() if end_frame_id is None else end_frame_id

        self.style_code_dim = 512
        self.style_mean_list = []
        test_data_loader = CheckpointDataLoader(self.test_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=False,
                                                     distributed=self.options.distributed)
        with torch.no_grad():
            for i in range(self.train_ds.first_frame, self.train_ds.last_frame):
                idx = torch.zeros(1).cuda() + i
                idx = idx.long()
                self.style_mean_list.append(self.style_code(idx).half())
        self.style_mean = torch.stack(self.style_mean_list)
        self.style_mean = torch.mean(self.style_mean, dim=0)
        self.test_ds.random_crop = False
        self.test_ds.training = False
        
        idx = torch.zeros(1).cuda().long()
        self.style_code_no = self.style_mean
        input_batch_list = []
        for step, input_batch in enumerate(tqdm.tqdm(test_data_loader, desc='Load data from disk',
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=0,
                                              ascii=True),
                                         test_data_loader.checkpoint_batch_idx):
            
            if step % 1 != 0:
                continue
            if self.distributed:
                if self.local_rank != 0:
                    return
            input_batch_list.append(input_batch)
        import time
        for i in tqdm.trange(len(input_batch_list)):
            input_batch = input_batch_list[i]
            for i in range(len(input_batch)):
                input_batch[i] = {k: v.half().to(self.device)
                            if isinstance(v, torch.Tensor) else v for k, v in input_batch[i].items()}
            bat_loc = len(input_batch) - 1
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                
                #cond = self.get_cond(input_batch[0])
                time_s = time.time()
                cond = self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond(input_batch[0])) + self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond2(input_batch[1]))
                img_pred = self.gen(input_batch[1], cond, idx)
                time_e = time.time()
                # print(time_e - time_s)
            

            img_fname = input_batch[bat_loc]['frame_fname'][0]
            out_path_rgb = os.path.join(out_dir_rgb, img_fname)

            sample = img_pred
            sample = sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            sample = np.uint8(np.clip(sample*127.5+127.5, 0, 255))
            cv.imwrite(out_path_rgb, sample[:, :, ::-1])

    def test_seq2(self, start_frame_id=None, end_frame_id=None, resume=False, cam_ids=None):
        out_dir_rgb = os.path.join(self.options.log_dir, 'step_%d' % self.step_count, 'reconstruction_rgb')
        os.makedirs(out_dir_rgb, exist_ok=True)
        self.set_eval()
        self.triplane = self.triplane.half()
        self.triplane_conv = self.triplane_conv.half()
        self.diffnet = self.diffnet.half()
        start_frame_id = 0 if start_frame_id is None else start_frame_id
        end_frame_id = self.test_ds.frame_num() if end_frame_id is None else end_frame_id

        self.style_code_dim = 512
        self.style_mean_list = []
        test_data_loader = CheckpointDataLoader(self.test_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=False,
                                                     distributed=self.options.distributed)
        with torch.no_grad():
            for i in range(self.train_ds.first_frame, self.train_ds.last_frame):
                idx = torch.zeros(1).cuda() + i
                idx = idx.long()
                self.style_mean_list.append(self.style_code(idx).half())
        self.style_mean = torch.stack(self.style_mean_list)
        self.style_mean = torch.mean(self.style_mean, dim=0)
        self.test_ds.random_crop = False
        self.test_ds.training = False
        
        idx = torch.zeros(1).cuda().long()
        self.style_code_no = self.style_mean
        input_batch_list = []
        for step, input_batch in enumerate(tqdm.tqdm(test_data_loader, desc='Load data from disk2',
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=0,
                                              ascii=True),
                                         test_data_loader.checkpoint_batch_idx):
            
            if step % 1 != 0:
                continue
            if self.distributed:
                if self.local_rank != 0:
                    return
            # input_batch_list.append(input_batch)
            # input_batch = input_batch_list[i]
            for i in range(len(input_batch)):
                input_batch[i] = {k: v.half().to(self.device)
                            if isinstance(v, torch.Tensor) else v for k, v in input_batch[i].items()}
            bat_loc = len(input_batch) - 1
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                
                #cond = self.get_cond(input_batch[0])
                time_s = time.time()
                cond = self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond(input_batch[0])) + self.triplane_conv(self.get_cond(input_batch[1]), self.get_cond2(input_batch[1]))
                img_pred = self.gen(input_batch[1], cond, idx)
                time_e = time.time()
                print(time_e - time_s)
            

            img_fname = input_batch[bat_loc]['frame_fname'][0]
            # import pdb
            # pdb.set_trace()
            img_gt = input_batch[bat_loc]['img']
            smpl_map = input_batch[bat_loc]['smpl_map']
            out_path_rgb = os.path.join(out_dir_rgb, img_fname)

            sample = img_pred
            sample = sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            sample = np.uint8(np.clip(sample*127.5+127.5, 0, 255))
            img_gt = img_gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_gt = np.uint8(np.clip(img_gt*127.5+127.5, 0, 255))
            smpl_map = smpl_map[0].permute(1, 2, 0).detach().cpu().numpy()
            smpl_map[:, :, 1] -= 0.4
            #smpl_map = (smpl_map / 255.0 - 0.5) / 0.3
            smpl_map = (smpl_map * 0.3 + 0.5) * 255.0

            skel_map = input_batch[bat_loc]['hand_skel_map'][0].permute(1, 2, 0).detach().cpu().numpy()
            #smpl_map = (smpl_map / 255.0 - 0.5) / 0.3
            skel_map = skel_map * 255.0
            skel_map[...,2] += skel_map[...,3]
            skel_map[...,2] += skel_map[...,4]
            skel_map = skel_map[...,:3]
            sample = np.concatenate([img_gt, sample, smpl_map, skel_map], axis=1)
            cv.imwrite(out_path_rgb, sample[:, :, ::-1])
        
     
            