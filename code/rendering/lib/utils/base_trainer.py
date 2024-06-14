from __future__ import division
import os, sys
import time, datetime
import logging
import numpy as np
import random
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .data_loader import CheckpointDataLoader
from .saver import CheckpointSaver
from .general import configure_logging, create_code_snapshot, to_torch_device


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed+worker_id)


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.endstep = self.options.end_step
        configure_logging(self.options.debug, self.options.quiet, self.options.logfile)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.dataset_perm = None
        self.models_dict = None
        self.optimizers_dict = None
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(
                self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

        logging.info('Starting epoch count: %d' % self.epoch_count)
        logging.info('Starting step count: %d' % self.step_count)

        # backup code
        now = datetime.datetime.now()
        code_bk_path = os.path.join(
            self.options.log_dir, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
        create_code_snapshot('./', code_bk_path,
                             extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh', '.txt'),
                             exclude=('examples', 'third-party', 'bin', 'logs', 'results', 'debug'))

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    logging.info('Pretrained checkpoint loaded for %s' % model)

    def train(self):
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        """Training process."""
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch
            self.epoch_count = epoch
            self.start_epoch(epoch)
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
                                                     worker_init_fn=worker_init_fn,distributed=self.options.distributed)

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx,
                                              ascii=True),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime and self.step_count < self.endstep:
                    '''
                    
                    '''
                    if isinstance(batch, list):
                        for i in range(len(batch)):   
                            batch[i] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch[i].items()}
                    else:
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, out)

                    # Backup the current training stage
                    if self.step_count % (self.options.summary_steps*50) == 0:
                        self.saver.save_latest(
                            self.models_dict, self.optimizers_dict, epoch, step + 1,
                            self.options.batch_size, None,
                            self.step_count)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(
                            self.models_dict, self.optimizers_dict, epoch, step+1,
                            self.options.batch_size, None,
                            self.step_count)
                        '''
                        self.saver.save_checkpoint(
                            self.models_dict, self.optimizers_dict, epoch, step+1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count)
                        '''
                        logging.info('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    logging.info('Timeout reached')
                    self.saver.save_checkpoint(
                        self.models_dict, self.optimizers_dict, epoch, step,
                        self.options.batch_size, train_data_loader.sampler.dataset_perm,
                        self.step_count)
                    logging.info('Checkpoint saved')
                    return

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            # if (epoch+1) % 10 == 0:
            #     # self.saver.save_checkpoint(
            #     # self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
            #     self.saver.save_checkpoint(
            #         self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size,
            #         None, self.step_count)
            self.end_epoch(epoch)
        return

    def generate_results(self):
        """Inference process."""
        train_data_loader = CheckpointDataLoader(self.test_ds, checkpoint=None,
                                                 dataset_perm=None,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 pin_memory=self.options.pin_memory,
                                                 shuffle=False,
                                                 worker_init_fn=None)

        # Iterate over all batches in an epoch
        for step, batch in enumerate(tqdm(train_data_loader, desc='Inference',
                                          total=len(self.test_ds),
                                          initial=train_data_loader.checkpoint_batch_idx,
                                          ascii=True),
                                     train_data_loader.checkpoint_batch_idx):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.generate_results_step(batch)

        return

    # The following methods (with the possible exception of test)
    # have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        pass

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def generate_results_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch, losses=None):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass
