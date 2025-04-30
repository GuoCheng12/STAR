import pdb
import argparse
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from libs.builders import build_trainer, build_tester, build_models, build_dataloaders
from libs.utils import init_all


def main():
    parser = argparse.ArgumentParser(description='AstroIR -- Dawn of Starbase-10K')
    parser.add_argument('config', type=str)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')


    parser.add_argument('--use_loss', type=str, default=None, 
                        help='Override the use_loss parameter in the config file (e.g., "L1", "L2")')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Set use_att to True if this flag is provided')
    args = parser.parse_args()
    configs, logger = init_all(args)


    if args.use_loss is not None:
        configs['model']['use_loss'] = args.use_loss
    if args.use_attention:
        configs['model']['use_attention'] = True
        
    trainloader, evalloader = build_dataloaders(**configs['dataset'])
    # pdb.set_trace()
    net = build_models(logger, **configs['model']).to('cuda')
    if args.evaluate:
        tester = build_tester(net, evalloader, **configs['test'])
        tester.test()
        print('done')
        exit()

    trainer = build_trainer(net, logger, trainloader, evalloader, **configs['train'])
    trainer.train()


if __name__ == "__main__":
    main()
