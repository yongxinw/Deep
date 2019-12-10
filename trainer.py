from model import DeepVP
from dataloader import get_dataloader
from utils import time_for_file, print_log, AverageMeter, device

import torch
import torch.nn as nn

from tqdm import tqdm
import tensorboardX
import os.path as osp
import os
import time
import datetime


class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.train_loader, self.val_loader = self.setup_data()
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        self.loss_func = self.setup_loss()
        self.model_save_dir, self.tb_writer, self.val_tb_writer, self.log_file = self.setup_logs()

    def setup_model(self):
        model = DeepVP(resnet_pretrained=self.config.resnet_pretrained,
                       resnet_usebn=self.config.resnet_usebn,
                       grid_resolution=self.config.grid_resolution,
                       resnet_depth=self.config.resnet_depth).to(device)
        return model

    def setup_optimizer(self):
        optim = torch.optim.Adam(params=self.model.parameters(),
                                 lr=float(self.config.lr),
                                 weight_decay=float(self.config.weight_decay))

        return optim

    def setup_loss(self):
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.config.bce_pos_weight])).to(device)
        return criterion

    def setup_data(self):
        train_loader, val_loader = get_dataloader(image_dir=self.config.image_dir,
                                                  label_path=self.config.label_path,
                                                  grid_resolution=self.config.grid_resolution,
                                                  batch_size=self.config.batch_size,
                                                  num_workers=self.config.num_workers,
                                                  shuffle=self.config.shuffle,
                                                  dummy=True)
        return train_loader, val_loader

    def setup_logs(self):
        """
        Set up log directory and loggers
        :return:
        """
        exp_dir = osp.join(self.args.log_dir, self.args.experiment_name)
        model_save_dir = osp.join(exp_dir, "checkpoints")
        tb_log_dir = osp.join(exp_dir, "tb_logs")
        image_save_dir = osp.join(exp_dir, "images")

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(tb_log_dir, exist_ok=True)
        os.makedirs(image_save_dir, exist_ok=True)

        tb_writer = tensorboardX.SummaryWriter(osp.join(tb_log_dir, 'train'))
        val_tb_writer = tensorboardX.SummaryWriter(osp.join(tb_log_dir, 'val'))
        log_file = open(osp.join(exp_dir, "logs-{}.log".format(time_for_file())), 'w')

        return model_save_dir, tb_writer, val_tb_writer, log_file

    def save_model(self, ep):
        """
        Save model
        :param ep:
        :return:
        """
        save_path = osp.join(self.model_save_dir, "net_{:02d}.pth".format(ep))
        save_opt_path = osp.join(self.model_save_dir, "opt_{:02d}.pth".format(ep))
        torch.save(self.model.state_dict(), save_path)
        torch.save(self.optimizer.state_dict(), save_opt_path)

    def log(self, loss_dict, it, pbar, elapsed, val=False):
        """Logger to print results and stuff"""
        if not val:
            print_str = "Elapsed {:s}, iteration {:06d}:".format(elapsed, it)
            writer = self.tb_writer
        else:
            print_str = "Validation - Elapsed {:s}, iteration {:06d}:".format(elapsed, it)
            writer = self.val_tb_writer

        for name, item in loss_dict.items():
            # print_str += "\t{:s} {:.4f}".format(name, val)
            text, value = item
            print_str += text
            writer.add_scalar(name, value, global_step=it)

        print_log(print_str, log=self.log_file, pbar=pbar)

    def train(self):
        """
        Training method
        :return:
        """
        ep_pbar = tqdm(range(self.config.max_epochs))

        # to train mode
        self.model.train()
        print_log(self.model.__str__(), log=self.log_file, pbar=ep_pbar)

        # keeping the losses for log
        loss_dict = {}
        Loss = AverageMeter()

        # running time
        global_time = time.time()

        for ep in ep_pbar:
            for batch, data in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                loss = self.forward(data)
                loss.backward()
                self.optimizer.step()

                Loss.update(loss.item())
                loss_dict['Loss/total_loss'] = ("\t{:s} {:.4f} ({:.4f})".format("total_loss", loss.item(), Loss.avg),
                                                Loss.avg)

                if batch % self.config.log_every_iter == 0:
                    it = ep * len(self.train_loader) + batch
                    elapsed = int(time.time() - global_time)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    self.log(loss_dict, it, ep_pbar, elapsed)
                    Loss.reset()

            if ep % self.config.save_every_epoch == 0:
                it = (ep + 1) * len(self.train_loader)
                self.save_model(ep)

                self.model.eval()
                self.val(it, ep_pbar)
                self.model.train()

    def val(self, it, pbar):
        """
        Validation method
        :param it:
        :param pbar:
        :return:
        """
        # keeping the losses for log
        loss_dict = {}
        Loss = AverageMeter()
        global_time = time.time()

        for batch, data in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                loss = self.forward(data)
            Loss.update(loss.item())

        loss_dict['Loss/total_loss'] = ("\t{:s} {:.4f} ({:.4f})".format("total_loss", Loss.avg, Loss.avg), Loss.avg)
        elapsed = int(time.time() - global_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        self.log(loss_dict, it, pbar, elapsed=elapsed, val=True)

    def forward(self, data):
        """
        A single forward pass
        :param data:
        :return:
        """
        images = data['image'].to(device)
        labels = data['grid_label'].to(device)
        preds = self.model(images)
        loss = self.loss_func(preds, labels)
        return loss
