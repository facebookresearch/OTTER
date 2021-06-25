# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.wit import WIT
from data.dataset_factory import OpenImageDataset
from data.dataset_factory import ConceptualCaptions
from data.dataset_factory import padded_collate
from data.dataset_factory import DATASET_PATHS

from data.yfcc import YFCC, yfcc_transform, yfcc_collate


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


def build_transforms(args):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))

    if args.image_aug:
        train_trans = [
            transforms.RandomResizedCrop(224, ratio=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        train_trans = [
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize
        ]
    val_trans = [
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = transforms.Compose(train_trans)
    val_transform = transforms.Compose(val_trans)
    return train_transform, val_transform


def build_yfcc_loader(args, train=True):
    train_set = YFCC(
        Path(args.data_root, DATASET_PATHS[args.dataset]),
        transform=yfcc_transform(train=train),
    )

    # Turn off shuffling on YFCC, because it seems to cause Too Many Open Files Error.
    if args.distributed:
        train_sampler = DistributedSampler(train_set, shuffle=False)
        loader = DataLoader(
            train_set,
            batch_size=args.batch,
            num_workers=args.num_workers,
            drop_last=True,
            persistent_workers=True,
            sampler=train_sampler,
            collate_fn=yfcc_collate,
            worker_init_fn=set_worker_sharing_strategy
        )
    else:
        loader = DataLoader(
            train_set,
            batch_size=args.batch,
            num_workers=args.num_workers,
            drop_last=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=yfcc_collate,
            worker_init_fn=set_worker_sharing_strategy
        )
    return loader


def build_loaders(args):
    train_transform, val_transform = build_transforms(args)

    val_set = OpenImageDataset(
        root=Path(args.data_root, DATASET_PATHS["open-images"]),
        transform=val_transform,
    )
    val_loader = DataLoader(val_set, batch_size=512, num_workers=args.num_workers, shuffle=False,
                            collate_fn=padded_collate)

    # YFCC needs to be handled specially.
    if args.dataset == "yfcc":
        train_loader = build_yfcc_loader(args, train=True)
        return train_loader, val_loader

    if args.dataset == "wit":
        train_set = WIT(
            Path(args.data_root, DATASET_PATHS[args.dataset]),
            transform=train_transform,
        )
    elif args.dataset == "cc":
        train_set = ConceptualCaptions(
            Path(args.data_root, DATASET_PATHS[args.dataset]),
            transform=train_transform,
        )

    if args.distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch,
            num_workers=args.num_workers,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        )

    return train_loader, val_loader
