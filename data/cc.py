import io
import os

from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pil_loader(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image


class ConceptualCaptions(Dataset):
    def __init__(self, root_dir, transform=None, aug=None):
        self.root_dir = root_dir

        label_dir = os.path.join(self.root_dir, "processed_labels.csv")
        self.data_df = pd.read_csv(label_dir)

        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        j, filename, caption = self.data_df.iloc[idx]
        image_path = os.path.join(self.root_dir, filename)

        sample = pil_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.aug is not None:
            caption = self.aug.augment(caption)
        return sample, caption


def preprocess_cc(root_dir):
    """
    Not all the images in cc/cc12m dataset downloaded can be opened by PIL. This function attempts to read
    each image specified by the tsv file and filter out the ones that cannot be opened. The output is
    processed_labels.csv that will be required in ConceptualCaptions dataset above.
    """
    if 'cc12m' in root_dir:
        train_tsv = "cc12m.tsv"
        train_col_names = ["url", "caption"]
        download_col_names = ["caption", "filename", "split", "type", "size", "status", "url"]
    elif 'cc' in root_dir:
        train_tsv = "Train-GCC-training.tsv"
        train_col_names = ["caption", "url"]
        download_col_names = ["filename", "split", "type", "size", "status", "url"]

    caption_file = os.path.join(root_dir, train_tsv)
    download_file = os.path.join(root_dir, "downloaded_training_report.tsv")

    captions_df = pd.read_csv(caption_file, sep="\t", quotechar='"', names=train_col_names)
    download_df = pd.read_csv(download_file, sep="\t", quotechar='"', names=download_col_names)[
        ["filename", "url"]]
    data_df = captions_df.merge(download_df, on="url", how="inner")[["filename", "caption"]]
    invalid_indices = []
    for i in range(len(data_df)):
        if i % 50000 == 0:
            print("Loading {} / {}".format(i, len(data_df)))
        try:
            filename, caption = data_df.iloc[i]
            image_path = os.path.join(root_dir, filename)
            sample = pil_loader(image_path)
        except:
            invalid_indices.append(i)
    print("Number of invalid indices: {}".format(len(invalid_indices)))
    data_df = data_df.drop(index=invalid_indices)
    data_df.reset_index(drop=True, inplace=True)

    label_dir = os.path.join(root_dir, "processed_labels.csv")
    data_df.to_csv(label_dir)


if __name__ == "__main__":
    preprocess_cc('/data/data/cc')
