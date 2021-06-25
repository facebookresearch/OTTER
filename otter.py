# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import urllib.request
from PIL import Image
import torch
from torchvision import transforms
from models.model_factory import build_image_text_model

# ResNet50 + DeCLUTR-Sci-base model weights.
_MODELS = {
    "InfoNCE": "https://onedrive.live.com/download?cid=CDD071074C65025E&resid=CDD071074C65025E%21872&authkey=APyQBYUHUiU2voc",
    "LS": "https://onedrive.live.com/download?cid=CDD071074C65025E&resid=CDD071074C65025E%21874&authkey=AHo06qdL59Mx39M",
    "KD": "https://onedrive.live.com/download?cid=CDD071074C65025E&resid=CDD071074C65025E%21873&authkey=AAETynOUaaHM7jQ",
    "OTTER": "https://onedrive.live.com/download?cid=CDD071074C65025E&resid=CDD071074C65025E%21871&authkey=ANIpSxwJ3x9MAao",
}


def load(name, pretrained=True):
    assert name in _MODELS.keys(), f"Model name must be in {list(_MODELS.keys())}."

    model = build_image_text_model(
        "resnet50",
        "declutr-sci-base",
        embedding_dim=768,
        max_token_length=60,
        label_smoothing=False,
        pretrain=True,
        lock_image=False
    )

    if pretrained:
        url = _MODELS[name]
        path = f'./pretrained/{name}.pth.tar'

        if not os.path.exists('./pretrained/'):
            os.makedirs('./pretrained/')

        # Download checkpoint
        if not os.path.exists(path):
            print("Downloading model to ./pretrained")
            urllib.request.urlretrieve(url, path)
            print("Downloaded")

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
    return model, _transform()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temperature = 60

    model, preprocess = load("InfoNCE")
    model = model.to(device)

    image = Image.open("doge.jpg")
    image = preprocess(image).unsqueeze(0).to(device)
    texts = ['photo of a dog', 'photo of a sofa', 'photo of a flower']

    with torch.no_grad():
        features = model.forward_features(image, texts)
        image_logits, text_logits = model.compute_logits(features)
        image_logits *= temperature

        probs = image_logits.softmax(dim=-1).cpu().numpy()

    print("Probs:", probs)  # Probs: [[0.92657197 0.00180788 0.07162025]]
