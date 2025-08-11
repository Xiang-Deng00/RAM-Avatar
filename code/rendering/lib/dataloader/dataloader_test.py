import os
import glob
import warnings
import logging

import cv2 as cv
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import json
import copy
import tqdm
import random




def get_frame_id_offset(dataset_path):
    return 0
class Dataset(data.Dataset):
    def __init__(self, dataset_path, cam_num, cam_ids_to_use, smpl_params_fpath,
                 first_frame=0, last_frame=400, white_bkgd=False, resolution=512, training=True,
                 encode_smpl_map=True, config=None):
        
        self.base_dir = os.path.abspath(dataset_path)
        self.last_frame = last_frame
        self.first_frame = first_frame
        self.white_bkgd = white_bkgd
        self.cam_num = cam_num
        self.cam_ids_to_use = cam_ids_to_use
        self.ratio = 1
        self.default_near, self.default_far = 0.5, 3.0
        self.resolution = resolution
        self.smpl_params_fpath = smpl_params_fpath
        self.training = training
        self.random_crop = training
        self.encode_smpl_map = encode_smpl_map
        self.frame_id_offset = get_frame_id_offset(self.base_dir)
        self.h, self.w = 3840, 2160
        self.config = config
        self.bg = np.zeros([self.h, self.w, 3], dtype=np.uint8)
        self.cond_ebd_freq_num = 4

        self.adaptive_sample = True
        self.t_length = self.config.t_length
        self.face = self.config.face


    def frame_num(self):
        return (self.last_frame - self.first_frame + 1)

    def view_num(self):
        return len(self.cam_ids_to_use)

    def cond_channel_num(self):
        return (self.cond_ebd_freq_num*2+1) * 3 + 1

    def __len__(self):
        return (self.last_frame - self.first_frame + 1) * len(self.cam_ids_to_use)


    def get_item(self, frame_id):
        l1 = self.config.l1
        r1 = self.config.r1
        leng = self.config.leng
        border = 500
        
        if os.path.isdir(self.smpl_params_fpath):
            smpl_params_folder = self.smpl_params_fpath
        else:
            smpl_params_folder = os.path.dirname(self.smpl_params_fpath)
        # print('smpl_params_fpath:', self.smpl_params_fpath, smpl_params_folder)
        subscr = 100
        smpl_map_fpath = os.path.join(smpl_params_folder, '%08d.png' % (frame_id - subscr))
        smpl_map_fpath2 = os.path.join(smpl_params_folder + '_001', '%08d.png' % (frame_id - subscr))
        # print('smpl:', smpl_map_fpath, smpl_map_fpath2)
        smpl_map = cv.imread(smpl_map_fpath, cv.IMREAD_UNCHANGED)
        smpl_map = cv.copyMakeBorder(smpl_map, border, border, border, border, cv.BORDER_CONSTANT, value=0)
        smpl_map2 = cv.imread(smpl_map_fpath2, cv.IMREAD_UNCHANGED)
        smpl_map2 = cv.copyMakeBorder(smpl_map2, border, border, border, border, cv.BORDER_CONSTANT, value=0)
        hand_kpt_fpath = os.path.join(self.base_dir, 'keypoints_mmpose_hand/%08d.json' % (frame_id))
        skel_map = self.gen_hand_skeleton_map(smpl_map.shape[0], smpl_map.shape[1], hand_kpt_fpath)
        skel_map = cv.copyMakeBorder(skel_map, border, border, border, border, cv.BORDER_CONSTANT, value=0)
        if self.face:
            face_map = os.path.join(self.base_dir, 'track2/%08d.png' % (frame_id))
            face = cv.imread(face_map)
            face = face[:,:,::-1]
            face = cv.copyMakeBorder(face, border, border, border, border, cv.BORDER_CONSTANT, value=0)

        smpl_map = smpl_map[l1:l1+leng, r1:r1+leng]
        smpl_map2 = smpl_map2[l1:l1+leng, r1:r1+leng]
        skel_map = skel_map[l1:l1+leng, r1:r1+leng]
        smpl_map = cv.resize(smpl_map, (self.resolution, self.resolution), interpolation=cv.INTER_NEAREST)
        smpl_map2 = cv.resize(smpl_map2, (self.resolution, self.resolution), interpolation=cv.INTER_NEAREST)
        skel_map = cv.resize(skel_map, (self.resolution, self.resolution), interpolation=cv.INTER_NEAREST)
        if self.face:
            face = face[l1:l1+leng, r1:r1+leng]
            face = cv.resize(face, (self.resolution, self.resolution), interpolation=cv.INTER_NEAREST)

        
        skel_map = np.float32(skel_map) / 255.0
        if self.face:
            face = np.float32(face) / 255.0

        if 0:
            smpl_map = np.float32(smpl_map) / 255.0
        else:
            smpl_map_msk = smpl_map[:, :, -1:] / 255.0
            smpl_map = smpl_map[:, :, :-1]
            smpl_map = (smpl_map / 255.0 - 0.5) / 0.3
            smpl_map[:, :, 1] += 0.4
            smpl_map = smpl_map * smpl_map_msk
            smpl_map = np.float32(smpl_map)

            
            smpl_map_msk2 = smpl_map2[:, :, -1:] / 255.0
            smpl_map2 = smpl_map2[:, :, :-1]
            smpl_map2 = (smpl_map2 / 255.0 - 0.5) / 0.3
            smpl_map2[:, :, 1] += 0.4
            smpl_map2 = smpl_map2 * smpl_map_msk2
            smpl_map2 = np.float32(smpl_map2)

        if self.encode_smpl_map:
            smpl_map = self.encode_condition(smpl_map)

        return_dict = {
            'idx': torch.tensor(frame_id - self.frame_id_offset, dtype=torch.long),
            'frame_id': frame_id,
            'frame_fname': '%08d.png' % (frame_id),
            'smpl_map': torch.from_numpy(smpl_map).permute(2, 0, 1),
            'smpl_map2': torch.from_numpy(smpl_map2).permute(2, 0, 1),
            'hand_skel_map': torch.from_numpy(skel_map).permute(2, 0, 1),
            'face': torch.from_numpy(face).permute(2, 0, 1),
        }
        if self.face:
            return_dict['face'] = torch.from_numpy(face).permute(2, 0, 1)
        return return_dict


    def __getitem__(self, item):
        if self.training and self.adaptive_sample:
            frame_id = np.random.choice(range(self.frame_num())) + self.first_frame
        else:
            frame_id = item // len(self.cam_ids_to_use)
            frame_id = self.first_frame + frame_id 
        return_dict_list = []


        for i in range(self.t_length - 1):
            return_dict_list.append(self.get_item(frame_id=frame_id-self.t_length + i + 1))
        return_dict_list.append(self.get_item(frame_id=frame_id))
        for i in range(self.t_length - 1):
            return_dict_list.append(self.get_item(frame_id=frame_id+self.t_length - i - 1))

        return return_dict_list


    def get_mask(self, img_fpath):
        """Follow the same preprocessing as NeuralBody"""
        img_fname = os.path.basename(img_fpath)
        msk_fpath = img_fpath.replace(img_fname, 'mask/pha/' + img_fname).replace('png', 'png')
        msk = cv.imread(msk_fpath, cv.IMREAD_UNCHANGED)
        return msk

    def gen_hand_skeleton_map(self, img_h, img_w, kpt_fpath):
        with open(kpt_fpath, 'r') as fp:
            kpts = json.load(fp)
        skel_map = np.zeros([5, img_h, img_w, 3], dtype=np.uint8)
        for kpt_one_hand in (kpts['lhand'][0], kpts['rhand'][0]):
            for i in range(5):
                skels = [kpt_one_hand[0], kpt_one_hand[4*i+1], kpt_one_hand[4*i+2],
                         kpt_one_hand[4*i+3], kpt_one_hand[4*i+4]]
                for jnt_i, jnt_j in zip(skels[:-1], skels[1:]):
                    jnt_i_ = np.round(np.asarray(jnt_i[:2])).astype(np.int32)
                    jnt_j_ = np.round(np.asarray(jnt_j[:2])).astype(np.int32)
                    skel_map[i] = cv.line(skel_map[i], jnt_i_, jnt_j_, (255, 255, 255), thickness=int(16/1))
        skel_map = np.transpose(skel_map[:, :, :, 0], (1, 2, 0))
        return skel_map

    def sample_hand_bbox(self, hand_skel_map, bbox_size=128):
        msk = np.sum(hand_skel_map, axis=-1)
        if np.sum(msk) < 1e-3: 
            rcenter = np.random.randint(bbox_size//2, hand_skel_map.shape[0]-(bbox_size-bbox_size//2))
            ccenter = np.random.randint(bbox_size//2, hand_skel_map.shape[1]-(bbox_size-bbox_size//2))
        else:
            rids, cids = np.nonzero(msk > 1e-3)
            i = np.random.randint(0, len(rids))
            rcenter = np.clip(rids[i], bbox_size//2, hand_skel_map.shape[0]-(bbox_size-bbox_size//2))
            ccenter = np.clip(cids[i], bbox_size//2, hand_skel_map.shape[1]-(bbox_size-bbox_size//2))
        bbox = [ccenter-bbox_size//2, rcenter-bbox_size//2,
                ccenter-bbox_size//2+bbox_size, rcenter-bbox_size//2+bbox_size]
        bbox = np.asarray(bbox)
        return bbox