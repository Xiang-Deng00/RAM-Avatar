import os
from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch
import trimesh
import logging
import tqdm
from tensorboardX import SummaryWriter


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def configure_logging(debug, quiet, logfile):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)

    trimesh_logger = logging.getLogger('trimesh')
    trimesh_logger.setLevel(logging.WARNING)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def load_point_cloud_by_file_extension(file_name):

    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    else:
        point_set = torch.tensor(trimesh.load(file_name, ext).vertices).float()

    return point_set


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)


def to_torch_device(data, device):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device)
    elif isinstance(data, Mapping):
        data = {k: to_torch_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        data = [to_torch_device(d, device) for d in data]
    elif isinstance(data, tuple):
        data = (to_torch_device(d, device) for d in data)
    # else:
    #     raise ValueError('Unsupported value type for transferring to device! ')
    return data


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def create_tensorboard_saver(experiment_dir):
    tensorboard_log_dir = os.path.join(experiment_dir, 'TensorboardLogs')
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    return SummaryWriter(tensorboard_log_dir)


def save_tensorboard_logs(saver, step, **kargs):
    if step % 10 == 0:
        for ln in kargs.keys():
            saver.add_scalar(ln, kargs[ln], step)


def get_spec_with_default(specs, key, default=None):
    keys = key.split('.')
    spec = specs
    for k in keys:
        try:
            spec = spec[k]
        except KeyError:
            return default
    return spec


def harden_mask(rgb, acc, white_bkgd, thres):
    if white_bkgd:
        rgb -= (1 - acc)
    acc_h = np.copy(acc)
    # acc_h[acc_h > thres] = 1
    # acc_h[acc_h <= thres] = 0
    acc_h = (acc_h - thres) * 4 + thres
    acc_h = np.clip(acc_h, 0, 1)

    rgb_h = rgb / (acc + 1e-12) * acc_h
    if white_bkgd:
        rgb_h += (1 - acc_h)
    return rgb_h, acc_h


def generate_rotating_cams(cam_data, render_pose_num):
    import cv2
    import copy

    cam_pos_all = np.zeros([len(cam_data), 3], dtype=np.float32)

    for cam_id in range(len(cam_data)):
        cam_R = np.array(cam_data[cam_id]['R']).astype(np.float32).reshape((3, 3))
        cam_t = np.array(cam_data[cam_id]['t']).astype(np.float32).reshape((3,))
        cam_RT = np.eye(4, dtype=np.float32)
        cam_RT[:3, :3] = cam_R
        cam_RT[:3, 3] = cam_t
        cam_RT = np.linalg.inv(cam_RT)
        cam_pos = cam_RT[:3, 3]
        cam_pos_all[cam_id] = cam_pos

    cam_center = np.mean(cam_pos_all, axis=0, keepdims=True)
    cam_pos_all -= cam_center
    u, s, vh = np.linalg.svd(cam_pos_all)
    rot_axis = vh[2]

    first_cam_data = cam_data[0]
    cam_R = np.array(first_cam_data['R']).astype(np.float32).reshape((3, 3))
    cam_t = np.array(first_cam_data['t']).astype(np.float32).reshape((3,))
    cam_RT = np.eye(4, dtype=np.float32)
    cam_RT[:3, :3] = cam_R
    cam_RT[:3, 3] = cam_t

    render_cams = []
    for i in range(render_pose_num):
        angle = 2*np.pi*i/render_pose_num
        aa = rot_axis * angle
        rot = cv2.Rodrigues(aa)[0]
        delta_RT = np.eye(4, dtype=np.float32)
        delta_RT[:3, :3] = rot
        delta_RT[:3, 3] = -np.matmul(cam_center, rot.transpose()) + cam_center
        new_RT = np.matmul(cam_RT, delta_RT)
        render_cam = copy.deepcopy(first_cam_data)
        render_cam['R'] = new_RT[:3, :3]
        render_cam['t'] = new_RT[:3, 3]
        render_cams.append(render_cam)
    return render_cams
