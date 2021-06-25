# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import urllib
import re

import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset

from datadings.reader import MsgpackReader
from datadings.torch import Dataset as DatadingDataset
from datadings.torch import CompressedToPIL
from datadings.torch import Compose


def yfcc_clean(x):
    html_re = re.compile('<.*?>')
    x = urllib.parse.unquote(str(x))
    x = x.replace("+", " ")
    x = re.sub(html_re, '', str(x))
    return x


def yfcc_transform(train=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711))
    if train:
        crop = transforms.RandomCrop(224)
    else:
        crop = transforms.CenterCrop(224)

    t = {'image': Compose(
        CompressedToPIL(),
        transforms.Resize(256),
        crop,
        transforms.ToTensor(),
        normalize
    )}
    return t


def yfcc_collate(batch):
    # batch contains a list of tuples of structure (img, targets)
    data = torch.stack([b['image'] for b in batch], dim=0)
    caption = [yfcc_clean(b['title']) + ". " + yfcc_clean(b['description']) for b in batch]
    return data, caption


class YFCC(ConcatDataset):
    def __init__(self, root_dir, transform):
        image_dir = os.path.join(root_dir, "datadings")

        msgpack_datasets = []
        for filename in os.listdir(image_dir):
            if filename.endswith(".msgpack"):
                msgpack_path = os.path.join(image_dir, filename)
                reader = MsgpackReader(msgpack_path)
                ds = DatadingDataset(
                    reader,
                    transforms=transform
                )
                msgpack_datasets.append(ds)
        super().__init__(msgpack_datasets)
