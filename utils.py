# Created by yongxinwang at 2019-12-08 18:41
import yaml
from easydict import EasyDict
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = EasyDict(cfg)
    cfg_str = open(cfg_path, 'r').read().splitlines()
    return cfg, cfg_str


def time_for_file():
    ISOTIMEFORMAT = '%h-%d-at-%H-%M'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))


def print_log(message=None, log=None, pbar=None):
    if message is None:
        message = ""
    if pbar is not None:
        pbar.write(message)
    else:
        print(message)

    if log is not None:
        log.write("{}\n".format(message))
        log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
