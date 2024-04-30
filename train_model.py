import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import construct_dataset


class M2depth:
    def __init__(self, cfg, rank):
        # super(M2depth, self).__init__(cfg)
        self.dataloaders = {}
        self.rank = rank
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        # self.models = self.prepare_model(cfg, rank)
        # self.losses = self.init_losses(cfg, rank)
        # self.view_rendering, self.pose = self.init_geometry(cfg, rank)
        # self.set_optimizer()

        # if self.pretrain and rank == 0:
        #     self.load_weights()

    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')

        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            # if rank == 0:
            #     self.set_val_dataloader(cfg)

        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)


    def set_train_dataloader(self, cfg, rank):
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas = self.world_size,
                rank=rank,
                shuffle=True
            )
            dataloader_opts['sampler'] = self.train_sampler

        self.dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_eval_dataloader(self, cfg):
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)