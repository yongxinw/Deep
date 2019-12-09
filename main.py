# Created by yongxinwang at 2019-12-08 17:25
import argparse

from trainer import Trainer
from utils import get_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for training Simultaneous Tracking and Detection with gnn")
    # TODO: add arguments
    parser.add_argument("--cfg_path", type=str, help="path of the cfg file")
    parser.add_argument("--log_dir", type=str, help="root directory for logging", required=True)
    parser.add_argument("--experiment_name", type=str, help="name of the current experiment", required=True)
    parser.add_argument("--resume", action="store_true", help="whether to resume from previous checkpoint")
    parser.add_argument("--model_resume_path", type=str, help="which model to resume from")

    # Parse args and load configs
    args = parser.parse_args()
    cfg, cfg_str = get_cfg(args.cfg_path)

    trainer = Trainer(args=args, config=cfg)

    # Train the network
    trainer.train()
