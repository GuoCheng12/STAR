import pdb
import math
import torch
from torch.utils.data import DataLoader

from .models.loss_function import *
from .models import MODEL
from .dataset import *
from .trainer import Trainer
from .tester import Tester
from .scheduler import Scheduler
from torch.utils.data import Subset
from torch import optim
from torch.utils.data import DistributedSampler as _DistributedSampler
 
import numpy as np


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
   
class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_losses(type,weight=1.0,**kargs):
    return general_loss(weight,globals()[type](**kargs))

def build_models(logger, type,checkpoint,**kargs):
    net = MODEL[type](**kargs)

    if checkpoint is not None:
        logger.info(f'{checkpoint} is loaded')
        checkpoint_data = torch.load(checkpoint)
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data
        new_state_dict = {}
        for key, value in state_dict.items():
            model_param = net.state_dict().get(key, None)
            if model_param is not None and model_param.shape == value.shape:
                new_state_dict[key] = value
            else:
                logger.info(f"Warning: Skipping parameter {key} due to shape mismatch or missing in the model.")
        net.load_state_dict(new_state_dict, strict=False)
    return net

def build_optimizer(model,optimizer):
    if 'sub_groups' not in optimizer: 
        optimizer = getattr(optim, optimizer.pop('type'))(model.parameters(), **optimizer)  
    else:
        for _ in optimizer['sub_groups']: _['params'] = getattr(model,_['params']).parameters()
        optimizer = getattr(optim, optimizer['type'])(optimizer['sub_groups']) 
    return optimizer 


def build_trainer(model, 
                 logger,
                 trainloader, 
                 evalloader,
                 optimizer,
                 scheduler,
                 **kargs):
    optimizer = build_optimizer(model,optimizer)
    scheduler_func = Scheduler(epoch_iter=len(trainloader), max_epoch=kargs['max_epoch'],**scheduler)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func)
    trainer = Trainer(model, optimizer, scheduler, trainloader, evalloader, logger, **kargs)
    return trainer

def custom_collate_fn(batch):
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:  # 如果批次为空，返回 None 或默认值
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def build_tester(model, evalloader, **kargs):

    tester = Tester(model, evalloader, **kargs)
    return tester

def build_dataloaders(type, batch_size, num_workers, ddp=False, local_rank=0, world_size=None, **kargs):
    trainset = globals()[type](split='train',**kargs)
    evalset = globals()[type](split='eval',**kargs)
    
    # trainset[0]
    # evalset = Subset(evalset, range(2000))
    if ddp:
        train_sampler = DistributedSampler(trainset, world_size, local_rank, shuffle=True)
        val_sampler = DistributedSampler(evalset, world_size, local_rank, shuffle=False)
    else:
        train_sampler, val_sampler = None, None

    trainloader = DataLoader(dataset=trainset,
               batch_size=batch_size,
               num_workers=num_workers,
               shuffle=(train_sampler is None),
               collate_fn=custom_collate_fn,
               pin_memory=True,
               drop_last=True,
               worker_init_fn=seed_worker,
               sampler=train_sampler,
               prefetch_factor=2,
               persistent_workers=True)
    evalloader = DataLoader(dataset=evalset,
               batch_size=batch_size,
               num_workers=num_workers,
               shuffle=False,
               pin_memory=True,
               drop_last=False,
               sampler=val_sampler,
               prefetch_factor=2,
               persistent_workers=True)
    return trainloader, evalloader