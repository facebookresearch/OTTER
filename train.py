# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.loader import build_loaders
from data.dataset_factory import get_metric
from models.model_factory import build_model
from optim.optim_factory import build_optimizer

import utils.utils as utils
from utils.checkpoint import save_model
from eval import test, train_knn_model


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser(description='OTTER training', add_help=False)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--output_dir', default='./out')

    # Model parameters
    parser.add_argument('--image_arch', default='resnet50', type=str)
    parser.add_argument('--text_arch', default='declutr-sci-base', type=str)
    parser.add_argument('--from_pretrained', action='store_true')
    parser.add_argument('--lock_image_encoder', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset', default='cc', type=str)
    parser.add_argument('--data_root', default='/home/ubuntu/data', type=str)
    parser.add_argument('--num_workers', default=16, type=int)

    # Augmentation parameters
    parser.add_argument('--image_aug', action="store_true")
    parser.add_argument('--text_aug', action="store_true")

    # printing params
    parser.add_argument('--print_freq', default=100, type=int)

    # Training parameters
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--max_token_length', default=60, type=int)

    # Distillation parameters
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema_distill', action='store_true')
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--distill_T_t', default=100.0, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--teacher_image_arch', default='', type=str)
    parser.add_argument('--teacher_text_arch', default='', type=str)
    parser.add_argument('--teacher_checkpoint', default='', type=str)

    # OT
    parser.add_argument('--ot_distill', action='store_true')
    parser.add_argument('--sinkhorn_lambda', default=0.1, type=float)
    parser.add_argument('--sinkhorn_iter', default=5, type=int)
    parser.add_argument('--vv_coef', default=1.0, type=float)
    parser.add_argument('--tt_coef', default=1.0, type=float)
    parser.add_argument('--global_ot', action='store_true')
    parser.add_argument('--remove_diag', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')

    # resume parameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    return parser


def train(loader, model, optimizer, device):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    end = time.time()
    model.train()

    for batch_idx, (images, texts) in enumerate(loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        losses = model(images, texts)

        loss = 0
        for _, v in losses.items():
            loss += v
        loss_meter.update(loss.item(), images.size(0))

        assert not (torch.isnan(loss) or torch.isinf(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % args.print_freq == 0:
            log = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | T: {T}'.format(
                batch=batch_idx + 1,
                size=len(loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=loss_meter.avg,
                T=model.module.get_temperature_str()
            )
            for k, v in losses.items():
                log += f" | {k}: {v:.4f}"
            print(log)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    train_loader, val_loader = build_loaders(args)

    model = build_model(args)
    model.to(device)

    optimizer = build_optimizer(args, model)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        del checkpoint

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True,
        )
    else:
        model = nn.DataParallel(model)

    best_prec1 = 0
    best_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_scheduler.get_last_lr()[0]))

        train(train_loader, model, optimizer, device)
        print("Evaluating Epoch: ", epoch + 1)

        if utils.is_main_process():
            prec1 = run_eval(model, val_loader, device)
            if prec1 > best_prec1:
                best_prec1 = prec1
                is_best = True
                best_epoch = epoch
            else:
                is_best = False
            save_model(args, epoch, model, optimizer, lr_scheduler, is_best)

        lr_scheduler.step()
        print(f"Best top1 acc {best_prec1:.4f} @ epoch {best_epoch+1}.")


def get_encoders(args, model):
    if args.distill:
        image_encoder = model.module.student.image_encoder
        text_encoder = model.module.student.text_encoder
    else:
        image_encoder = model.module.image_encoder
        text_encoder = model.module.text_encoder
    return image_encoder, text_encoder


def run_eval(model, val_loader, device):
    image_encoder, text_encoder = get_encoders(args, model)
    knn_model = train_knn_model(
        "open-images",
        text_encoder,
        dim=args.embedding_dim,
        use_templates=False,
        use_faiss=False,
    )

    prec1, output_str = test(
        val_loader,
        image_encoder,
        knn_model,
        get_metric("open-images"),
        device,
    )
    print("Accuracy: ", output_str)
    return prec1


if __name__ == "__main__":
    parser = argparse.ArgumentParser('OTTER training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
