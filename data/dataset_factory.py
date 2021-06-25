import os
from PIL import Image
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.metrics import Accuracy, FAtK


curr_dir = os.getcwd()
META_ROOT = os.path.join(curr_dir, "data/dataset_meta_data")

TARGET_DATASETS = {
    "open-images": "Custom",
    "ImageNet": "Default",
    "ImageNet21k": "Custom",
    "ImageNet22k": "Custom",
    "tencent": "Custom",
}

DATASET_PATHS = {
    "open-images": "open-images",
    "ImageNet": "imagenet/val",
    "ImageNet21k": "imagenet21k",
    "cc": "cc",
    "tencent": "tencent",
    "wit": "wit",
    # Add your dataset paths here.
}


class OpenImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.labeled_images = []

        img_idx_file = os.path.join(META_ROOT, "goi_images.txt")
        with open(img_idx_file, "r") as f:
            for l in f.readlines():
                entries = l.split("\t")
                img_name = entries[0]
                labels = [int(x) for x in entries[1:]]
                self.labeled_images.append([os.path.join(self.root, "test", img_name), labels])

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, idx):
        img_path, label_indices = self.labeled_images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label_indices


class ImageNet21kDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        with open(os.path.join(META_ROOT, "imagenet21k_wordnet_ids.txt"), "r") as f:
            wordnet_ids = [l.strip() for l in f.readlines()]

        with open(os.path.join(META_ROOT, "imagenet21k_wordnet_lemmas.txt"), "r") as f:
            wordnet_classes = [l.strip().replace("_", " ") for l in f.readlines()]
        assert len(wordnet_ids) == len(wordnet_classes)

        self.i2c = {wordnet_ids[i]: (i, wordnet_classes[i]) for i in range(len(wordnet_ids))}
        self.image_names = os.listdir(os.path.join(root, "images"))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        class_id = img_name.split("_")[0]
        label_index = self.i2c[class_id][0]
        img_path = os.path.join(self.root, "images", img_name)
        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        # Returning label_index as a list to be consistent with OpenImages
        return sample, [label_index]


class ImageNet22kDataset(Dataset):
    def __init__(self, root_21k, root_1k, transform=None):
        self.imagenet21k_dataset = ImageNet21kDataset(root=root_21k, transform=transform)
        self.imagenet21k_offset = len(get_classes("ImageNet21k"))
        self.imagenet1k_dataset = ImageFolder(root=root_1k, transform=transform)

    def __len__(self):
        return len(self.imagenet1k_dataset) + len(self.imagenet21k_dataset)

    def __getitem__(self, idx):
        if idx < len(self.imagenet21k_dataset):
            return self.imagenet21k_dataset.__getitem__(idx)
        else:
            sample, label_index = self.imagenet1k_dataset.__getitem__(idx - len(self.imagenet21k_dataset))
            label_index += self.imagenet21k_offset
            return sample, [label_index]


class TencentDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.labeled_images = []

        img_idx_file = os.path.join(META_ROOT, "tencent_val_image_id_from_imagenet.txt")
        with open(img_idx_file, "r") as f:
            for l in f.readlines():
                entries = l.split("\t")
                img_name = entries[0].split("/")[1]
                labels = [int(x) for x in entries[1:]]
                self.labeled_images.append([os.path.join(self.root, "images", img_name), labels])

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, idx):
        img_path, label_indices = self.labeled_images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(f)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label_indices


def build_datasets(data_root, datasets, transform):
    results = {}
    for name in datasets:
        assert name in TARGET_DATASETS.keys()

        dataset_path = Path(data_root, DATASET_PATHS[name])
        if name == "open-images":
            dataset = OpenImageDataset(root=dataset_path, transform=transform)
        elif name == "tencent":
            dataset = TencentDataset(root=dataset_path, transform=transform)
        elif name == "ImageNet":
            dataset = ImageFolder(root=dataset_path, transform=transform)
        elif name == "ImageNet21k":
            dataset = ImageNet21kDataset(root=dataset_path, transform=transform)
        elif name == "ImageNet22k":
            dataset = ImageNet22kDataset(
                root_21k=Path(data_root, DATASET_PATHS["ImageNet21k"]),
                root_1k=Path(data_root, DATASET_PATHS["ImageNet"]),
                transform=transform,
            )
        else:
            raise NotImplementedError("{} is not available".format(name))

        results[name] = dataset
    return results


def padded_collate(batch):
    # batch contains a list of tuples of structure (img, targets)
    data = torch.stack([item[0] for item in batch], dim=0)
    targets = [torch.LongTensor(item[1]) for item in batch]
    targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    return data, targets


def _get_imagenet_classes():
    with open(os.path.join(META_ROOT, "imagenet1k_wordnet_lemmas.txt"), "r") as f:
        # Chain together all the class labels with or.
        classes = [",or ".join([c.replace("_", " ").strip() for c in l.split(", ")]) for l in f.readlines()]
    return classes


def _get_tencent_classes():
    classes = []
    with open(os.path.join(META_ROOT, "tencent_classes.txt"), "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            cat_name = l.split("\t")[-1]
            # Chain together all the class labels with or
            classes.append(",or ".join([c.strip() for c in cat_name.split(", ")]))
    return classes


def _get_open_images_classes():
    with open(os.path.join(META_ROOT, "goi_classes.txt"), "r") as f:
        classes = [l.strip() for l in f.readlines()]
    return classes


def _get_imagenet21k_classes():
    with open(os.path.join(META_ROOT, "imagenet21k_wordnet_lemmas.txt"), "r") as f:
        # Chain together all the class labels with or.
        classes = [",or ".join([c.replace("_", " ").strip() for c in l.split(", ")]) for l in f.readlines()]
    return classes


def _get_imagenet21k_synsets():
    with open(os.path.join(META_ROOT, "imagenet21k_wordnet_ids.txt"), "r") as f:
        ids = [l.strip() for l in f.readlines()]
    return ids


def _get_imagenet22k_classes():
    imagenet21k_classes = _get_imagenet21k_classes()
    imagenet1k_classes = _get_imagenet_classes()
    return imagenet21k_classes + imagenet1k_classes


def get_classes(name):
    """
    Get a list of class names for a dataset.
    """
    if name == "open-images":
        return _get_open_images_classes()
    elif name == "tencent":
        return _get_tencent_classes()
    elif name == "ImageNet":
        return _get_imagenet_classes()
    elif name == "ImageNet21k":
        return _get_imagenet21k_classes()
    elif name == "ImageNet22k":
        return _get_imagenet22k_classes()
    else:
        raise NotImplementedError


def get_metric(name, K="1,2,5,10"):
    """
    Get the metric used to evaluate a dataset.
    """
    if name in ["open-images", "ImageNet21k", "ImageNet22k", "tencent"]:
        return FAtK(K)
    else:
        return Accuracy()


def get_templates(simple=False):
    if simple:
        return ['a photo of {}.']

    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.'
    ]
    return imagenet_templates
