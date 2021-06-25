from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.cc import ConceptualCaptions
from data.wit import WIT
from data.dataset_factory import OpenImageDataset
from data.dataset_factory import padded_collate
from data.dataset_factory import DATASET_PATHS


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


def build_loaders(args):
    train_transform, val_transform = build_transforms(args)

    val_set = OpenImageDataset(
        root=Path(args.data_root, DATASET_PATHS["open-images"]),
        transform=val_transform,
    )
    val_loader = DataLoader(val_set, batch_size=512, num_workers=args.num_workers, shuffle=False,
                            collate_fn=padded_collate)

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
