import os

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class WIT(Dataset):
    def __init__(self, root_dir, transform=None, aug=None):
        self.root_dir = root_dir

        label_dir = os.path.join(self.root_dir, "process_labels.csv")
        label_f = open(label_dir, "r")
        self.data_df = label_f.readlines()
        self.transform = transform
        self.aug = aug
        label_f.close()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        values = self.data_df[idx].split("\t")
        image_path = values[0]
        caption = "".join(values[1:])
        sample = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.aug is not None:
            caption = self.aug.augment(caption)
        return sample, caption
