# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import argparse
import pandas as pd
from PIL import Image


def pil_loader(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image


def preprocess_cc(root_dir, caption_tsv, download_report_tsv):
    """
    Not all the images in cc3m dataset downloaded can be opened by PIL. This function attempts to read
    each image specified by the tsv file and filter out the ones that cannot be opened. The output is
    processed_labels.csv that will be required in ConceptualCaptions dataset above.
    """
    train_col_names = ["caption", "url"]
    download_col_names = ["filename", "split", "type", "size", "status", "url"]

    caption_file = os.path.join(root_dir, caption_tsv)
    download_file = os.path.join(root_dir, download_report_tsv)

    captions_df = pd.read_csv(caption_file, sep="\t", quotechar='"', names=train_col_names)
    download_df = pd.read_csv(download_file, sep="\t", quotechar='"', names=download_col_names)[
        ["filename", "url"]]
    data_df = captions_df.merge(download_df, on="url", how="inner")[["filename", "caption"]]
    invalid_indices = []
    for i in range(len(data_df)):
        if i % 5000 == 0:
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


def preprocess_train(root_dir):
    preprocess_cc(root_dir,
                  "Train-GCC-training.tsv",
                  "downloaded_training_report.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conceptual Captions Preprocessing', add_help=False)
    parser.add_argument('--cc_root', default='/data/cc')
    args = parser.parse_args()
    preprocess_train(args.cc_root)
