# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import clip
import torch
from torch.utils.data import DataLoader

from data.dataset_factory import ConceptualCaptions, DATASET_PATHS
from data.yfcc import YFCC, yfcc_collate
from datadings.torch import CompressedToPIL, Compose
import utils.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='Dataset analysis', add_help=False)
    parser.add_argument('--data_root', default='/data/data/', type=str)
    parser.add_argument('--datasets', default='cc,yfcc', type=str)
    parser.add_argument('--batch', default=512, type=int)
    parser.add_argument('--stop', default=10, type=int)

    return parser


def on_diag_avg(input):
    return torch.mean(torch.diagonal(input, offset=0)).item()


def off_diag_avg(input):
    diag_mask = 1 - torch.eye(input.size(0)).to('cuda')
    return torch.mean(input * diag_mask).item()


def off_diag_max(input):
    diag_mask = 1 - torch.eye(input.size(0)).to('cuda')
    vals, indices = torch.max(input * diag_mask, dim=1)
    return torch.mean(vals).item()


def num_off_sig(input):
    diag_mask = 1 - torch.eye(input.size(0)).to('cuda')
    count = torch.sum((input * diag_mask) >= 0.2).item()
    return count


def remove_diagonal(input):
    n = input.size(0)
    return input.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)


def eval(model, loader, device):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    on_avg_meter = utils.AverageMeter()
    off_avg_meter = utils.AverageMeter()
    off_max_meter = utils.AverageMeter()
    off_sig_meter = utils.AverageMeter()

    end = time.time()
    model.eval()

    off_diag_sims = []
    for batch_idx, (images, texts) in enumerate(loader):
        data_time.update(time.time() - end)
        imgs = images.to(device)
        txts = torch.cat([clip.tokenize(c, truncate=True) for c in texts]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(imgs)
            text_features = model.encode_text(txts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        on_avg = on_diag_avg(similarity)
        off_avg = off_diag_avg(similarity)
        off_max = off_diag_max(similarity)
        num_sig = num_off_sig(similarity)

        off_diag_similarity = remove_diagonal(similarity).cpu()
        off_diag_sims.append(off_diag_similarity)

        on_avg_meter.update(on_avg)
        off_avg_meter.update(off_avg)
        off_max_meter.update(off_max)
        off_sig_meter.update(num_sig)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            log = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | on_avg: {on_avg:.3f} | ' \
                  'off_avg: {off_avg:.3f} | off_max: {off_max:.3f} |  off_sig: {off_sig:.3f}'.format(
                batch=batch_idx + 1,
                size=len(loader),
                data=data_time.avg,
                bt=batch_time.avg,
                on_avg=on_avg_meter.avg,
                off_avg=off_avg_meter.avg,
                off_max=off_max_meter.avg,
                off_sig=off_sig_meter.avg
            )
            print(log)

        if batch_idx == args.stop:
            break

    print('on_avg', on_avg_meter.avg)
    print('off_avg', off_avg_meter.avg)
    print('off_max', off_max_meter.avg)
    print('off_sig', off_sig_meter.avg)

    # Compute histogram
    off_diag_sim_matrix = torch.stack(off_diag_sims, dim=0)
    off_diag_sim_matrix = off_diag_sim_matrix.type(torch.float32)
    data = off_diag_sim_matrix.numpy().flatten()
    return data


def main(args):
    device = "cuda"
    model, preprocess = clip.load('ViT-B/32', device)

    datasets = args.datasets.split(",")

    for name in datasets:
        print(f"Evaluating {name}")
        collate_fn = None
        if name == "cc":
            dataset = ConceptualCaptions(Path(args.data_root, DATASET_PATHS[name]), transform=preprocess)
        elif name == "yfcc":
            compressed_to_pil = {'image': Compose(CompressedToPIL(), preprocess)}
            dataset = YFCC(Path(args.data_root, DATASET_PATHS[name]), transform=compressed_to_pil)
            collate_fn = yfcc_collate
        else:
            raise NotImplementedError

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch,
            num_workers=16,
            drop_last=True,
            persistent_workers=True,
            shuffle=True,
            collate_fn=collate_fn
        )

        data = eval(model, dataloader, device)
        plt.hist(data, density=True, bins=100)

    plt.yscale('log')
    plt.xlabel("Off-diag prob")
    plt.ylabel("percentage")
    plt.legend(datasets)
    plt.savefig(f"./off_diag_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('OTTER dataset analysis', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


# Results
# Batch 512 normalized
# CC:   on_diagonal_avg: 0.565, off_diagonal_avg: 0.001, off_diagonal_max: 0.215
# YFCC: on_diagonal_avg: 0.628, off_diagonal_avg: 0.001, off_diagonal_max: 0.197


# Batch 1024 normalized
# CC:   on_diagonal_avg: 0.480, off_diagonal_avg: 0.001, off_diagonal_max: 0.230
# YFCC: on_diagonal_avg: 0.551, off_diagonal_avg: 0.000, off_diagonal_max: 0.219

# Batch 2048 normalized
# CC:   on_diagonal_avg: 0.398, off_diagonal_avg: 0.000, off_diagonal_max: 0.238
# YFCC: on_diagonal_avg: 0.469, off_diagonal_avg: 0.000, off_diagonal_max: 0.239
