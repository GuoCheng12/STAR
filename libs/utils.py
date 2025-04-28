import torch
import numpy as np
import random
import os
import sys
import pdb
import logging

import math
from typing import Optional, Tuple
from importlib import import_module
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from io import BytesIO

import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from multiprocessing import Pool
import subprocess

import time as time

import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from torch import default_generator, randperm

from astropy.io import fits

from matplotlib.colors import Normalize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import sep


def init_logger(config):
    log_file = os.path.join(config['train']['log_dir'],'train.log')
    rank = config['train']['local_rank']
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def init_config(args):

    def fuse_config(base_cfg, cfg):
        for key in cfg.keys():
            if type(cfg[key]) is dict and type(base_cfg[key]) is dict:
                base_cfg[key] = fuse_config(base_cfg[key], cfg[key])
            else:
                if cfg.get('_delete_',False):
                    cfg.pop('_delete_')
                    return cfg
                else:
                    base_cfg[key] = cfg[key]
        return base_cfg
    def load_config(root_dir, cfg_name):
        sys.path.insert(0, root_dir)

        pyconfigs = import_module(cfg_name)
        configs = {}
        sys.path.pop(0)
        for key in dir(pyconfigs):
            if '__'!=key[:2]:
                component = getattr(pyconfigs,key)
                if type(component)==tuple: component = component[0]
                configs[key] = component
        if '_base_' in configs:
            base_infos = configs.pop('_base_').split('/')
            base_dir = "".join(base_infos[:-1])
            base_name = "".join(base_infos[-1:])
            base_root = os.path.join(root_dir,base_dir)
            base_config = load_config(base_root,base_name[:-3])
            configs = fuse_config(base_config, configs)
        return configs
    root_dir = os.path.abspath(os.path.dirname(args.config))
    cfg_name = args.config.split('/')[-1][:-3]
    
    configs = load_config(root_dir,cfg_name)

    if args.log_dir is not None:
        configs['train']['log_dir']=os.path.join('./log',args.log_dir)
    else:
        configs['train']['log_dir']=os.path.join('./log',args.config.split('configs')[-1][1:-3])
    os.makedirs(configs['train']['log_dir'],exist_ok=True)
    # configs['test']['visualize'] = args.visualize
    return configs

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)
    torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def backup_code(backup_lists, target_dir):
    target_dir = os.path.join(target_dir,'code_backup')
    os.makedirs(target_dir,exist_ok=True)
    for _ in backup_lists:
        source_dir = os.path.join(os.getcwd(),_)
        os.system('cp -r '+source_dir+' '+target_dir)
        print('Have backuped '+
              '\033[0;32;40mcontent\033[0m'.replace('content',_)+
              ' to '+
              '\033[0;35;40mcontent\033[0m'.replace('content',target_dir))

def init_ddp(args,configs): 
    configs['train']['local_rank'] = 0
    # configs['test']['local_rank'] = 0
    if args.launcher != 'none':
        _, configs['dataset']['local_rank'], configs['dataset']['world_size'] = init_dist_slurm(backend='nccl')
        # _, configs['dataset']['local_rank'], configs['dataset']['world_size'] = init_dist_pytorch(args.tcp_port, backend='nccl')
        configs['dataset']['ddp'] = True
        configs['train']['local_rank'] = configs['dataset']['local_rank']
        # configs['test']['local_rank'] = configs['dataset']['local_rank']
        configs['train']['ddp'] = True
        # configs['test']['ddp'] = True
    return configs

def init_all(args):
    configs = init_config(args)
    torch.set_default_dtype(getattr(torch,configs['dataset'].get('data_type','float32')))
    configs = init_ddp(args,configs)
    init_seed(configs['seed']+configs['train']['local_rank'])
    logger = init_logger(configs)
    logger.info('Running setting: {}'.format(args.config))
    if configs['train']['local_rank']==0:
        backup_lists = ['libs','tools','configs']
        backup_code(backup_lists, configs['train']['log_dir'])
    configs['model']['checkpoint'] = args.resume
    return configs, logger


def init_dist_slurm(tcp_port=None, backend='nccl'):
    num_gpus = torch.cuda.device_count()
    #rank = int(os.environ["SLURM_PROCID"])
    rank = int(os.environ["RANK"])
    #world_size = int(os.environ["SLURM_NTASKS"])
    world_size = int(os.environ["WORLD_SIZE"])
    #node_list = os.environ["SLURM_NODELIST"]
    #addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if tcp_port is not None:
        os.environ["MASTER_PORT"] = str(tcp_port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "32000"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    #os.environ["WORLD_SIZE"] = str(world_size)
    #os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    #os.environ["RANK"] = str(rank)

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return num_gpus, rank, world_size


def init_dist_pytorch(tcp_port, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank, world_size = get_dist_info()
    return num_gpus, rank, world_size

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def fits_vis(ori_array):
    from astropy.visualization import ZScaleInterval
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    z = ZScaleInterval()
    z1,z2 = z.get_limits(ori_array)
    norm = Normalize(vmin=z1, vmax=z2)
    normalized_array = norm(ori_array)
    cmap = plt.get_cmap('gray')
    wave_array = cmap(normalized_array)
    wave_array = (wave_array[..., 0]*255).astype(np.uint8)
    return wave_array


def vis_astro_SR(pred, target, input_img, mask, name, vis_dir):
    input_vis = fits_vis(np.squeeze(input_img))
    target_vis = fits_vis(np.squeeze(target))
    
    pred_masked = np.where(mask, pred, np.nan)
    pred_vis = fits_vis(np.squeeze(pred_masked))
    
    input_size = input_vis.shape[0]  # 例如 128
    hr_size = pred_vis.shape[0]      # 例如 256

    # 创建一个 1x3 的子图布局，宽度比例根据图像尺寸调整
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[input_size/hr_size, 1, 1])

    # 子图 1：低分辨率输入图像
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(input_vis, cmap='gray', extent=[0, input_size, 0, input_size])
    ax1.set_title('Input (LR)')
    ax1.set_aspect('equal')  # 保持宽高比
    ax1.axis('off')

    # 子图 2：预测高分辨率图像（应用掩码）
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(pred_vis, cmap='gray', extent=[0, hr_size, 0, hr_size])
    ax2.set_title('Prediction (HR)')
    ax2.set_aspect('equal')
    ax2.axis('off')

    # 子图 3：目标高分辨率图像
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(target_vis, cmap='gray', extent=[0, hr_size, 0, hr_size])
    ax3.set_title('Target (HR)')
    ax3.set_aspect('equal')
    ax3.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    os.makedirs(vis_dir, exist_ok=True) if not os.path.exists(vis_dir) else None  
    save_path = os.path.join(vis_dir, f"{name}_vis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


import lpips
# Assuming ssim and psnr are already imported (e.g., from skimage.metrics)

def evaluate_metric_SR(pred, target, mask):
    batch_ssim = 0.0
    batch_psnr = 0.0
    num_images = pred.size(0)

    for i in range(num_images):
        pred_i = pred[i].numpy().transpose(1, 2, 0)  
        target_i = target[i].numpy().transpose(1, 2, 0)
        mask_i = mask[i].numpy().transpose(1, 2, 0).astype(bool)
        pred_valid = pred_i[mask_i]
        target_valid = target_i[mask_i]

        if len(pred_valid) > 0:
            ssim_value = ssim(target_valid, pred_valid, 
                            data_range=target_valid.max() - target_valid.min(), 
                            multichannel=True)
            psnr_value = psnr(target_valid, pred_valid, 
                            data_range=target_valid.max() - target_valid.min())
            batch_ssim += ssim_value
            batch_psnr += psnr_value

    batch_ssim = batch_ssim / num_images if num_images > 0 else 0.0
    batch_psnr = batch_psnr / num_images if num_images > 0 else 0.0
    return batch_ssim, batch_psnr
