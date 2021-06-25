import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from utils.utils import AverageMeter
from data.loader import build_transforms
from data.dataset_factory import build_datasets, padded_collate, TARGET_DATASETS
from data.dataset_factory import get_classes, get_templates, get_metric
from models.model_factory import build_model
from utils.knn import FaissKNN
from sklearn.neighbors import KNeighborsClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


def get_args_parser():
    parser = argparse.ArgumentParser(description='ZSL eval', add_help=False)
    parser.add_argument('--batch', default=1024, type=int)
    parser.add_argument('-c', '--checkpoint', default='', type=str)

    # Model parameters
    parser.add_argument('--image_arch', default='resnet50', type=str)
    parser.add_argument('--text_arch', default='declutr-sci-base', type=str)
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('-K', default='1,2,5,10', type=str)
    parser.add_argument('--max_token_length', default=60, type=int)

    # data parameters
    parser.add_argument('--dataset', default="open-images", type=str)
    parser.add_argument('--data_root', default="/home/ubuntu/data", type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--use_templates', default=False, type=bool)

    return parser


def test(loader, model, knn_model, metric, device="cuda"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            # Compute the image embeddings and convert to numpy array
            image_embedding = model(inputs)["img_emb"]
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            image_embedding = image_embedding.cpu().numpy()

            # Convert target class ids to numpy array
            if isinstance(targets, list):
                targets = [t.cpu().numpy() for t in targets]
            else:
                targets = targets.cpu().numpy()

            # Use KNN to find TopK neighbors.
            if str(metric) == "FAtK":
                for k in metric.Ks:
                    predictions = knn_model.kneighbors(image_embedding, n_neighbors=k)
                    metric.update(targets, predictions, k)
            else:
                predictions = knn_model.predict(image_embedding)
                metric.update(targets, predictions)

        if batch_idx % 50 == 0:
            print("Batch {}, {}".format(batch_idx, metric.avg()))

        batch_time.update(time.time() - end)
        end = time.time()

    return metric.prec1, metric.avg()


def make_batches(tokens, batch_size):
    """
    Breaks a list into a list of lists each with batch_size elements.
    """
    if len(tokens) <= batch_size:
        return [tokens]
    else:
        batches = []
        while len(tokens) > 0:
            batches.append(tokens[:batch_size])
            tokens = tokens[batch_size:]
    return batches


def gen_caption_from_classes(class_names, templates):
    """
    Given a list of class names, return a list of template augmented captions,
    and the class_idx of these captions.
    captions: A list of strings describing each class
    labels_list: A list of ints representing the class index
    """
    captions = []
    labels_list = []
    for i, class_name in enumerate(class_names):
        if type(class_name) == str:
            class_name = [class_name]
        for c in class_name:
            for template in templates:
                caption = template.format(c)
                captions.append(caption)
                labels_list.append(i)
    return captions, labels_list


def get_dataset_embeddings(class_names, templates, text_model):
    """
    Compute the text embeddings of the class names of a dataset.
    :param class_names: List of class_names of a dataset
    :param templates: A list of potential templates to augment the class names.
    :param text_model: TextModel that computes the embedding of texts.
    :return: embedding_bank: (N,C), embedding_label: (N,)
    """
    captions, labels_list = gen_caption_from_classes(class_names, templates)
    text_model.eval()

    embedding_list = []
    with torch.no_grad():
        batches = make_batches(captions, 512)
        for captions_batch in batches:
            # Get the text embedding of the batch of class captions.
            embedding = text_model(captions_batch)["txt_emb"]
            embedding /= embedding.norm(dim=1, keepdim=True)
            embedding_list.append(embedding.detach().cpu())

    embedding_bank = torch.cat(embedding_list).numpy()
    embedding_labels = torch.LongTensor(labels_list)
    embedding_labels = embedding_labels.numpy()
    return embedding_bank, embedding_labels


def train_knn_model(dataset_name, text_model, dim, use_templates=False, use_faiss=True):
    class_names = get_classes(dataset_name)
    templates = get_templates(simple=(not use_templates))
    embedding_bank, embedding_labels = get_dataset_embeddings(class_names, templates, text_model)

    if use_faiss:
        knn_model = FaissKNN(n_neighbors=1, dim=dim)
    else:
        knn_model = KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=64)
    knn_model.fit(embedding_bank, embedding_labels)
    return knn_model


def main(args):
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset:
        target_datasets = args.dataset.split(",")
    else:
        target_datasets = TARGET_DATASETS.keys()
    print("target datasets: ", target_datasets)

    _, val_transform = build_transforms(args)
    datasets = build_datasets(
        datasets=target_datasets,
        transform=val_transform,
    )

    model = build_model(args)

    if args.checkpoint:
        load_checkpoint(model, args)

    image_model = nn.DataParallel(model.image_encoder)
    text_model = model.text_encoder

    image_model = image_model.to(device)
    text_model = text_model.to(device)

    results = {}
    for name, dataset in datasets.items():
        metric = get_metric(name, K=args.K)
        print(f"Evaluating: {name}, metric: {str(metric)}={args.K}")
        if name in ["open-images", "ImageNet21k", "ImageNet22k", "tencent"]:
            collate_fn = padded_collate
        else:
            collate_fn = default_collate

        val_loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.num_workers, shuffle=True,
                                collate_fn=collate_fn, drop_last=False)
        knn_model = train_knn_model(name, text_model, args.embedding_dim, use_templates=args.use_templates,
                                    use_faiss=True)
        prec1, output = test(val_loader, image_model, knn_model, metric, device)
        print(f"{name} | {str(metric)}: {output}")
        results[name] = output

    print(results)


def load_checkpoint(model, args):
    state_dict = torch.load(args.checkpoint)["state_dict"]

    # Get only the student state_dict
    copy_state_dict = state_dict.copy().items()
    for name, v in copy_state_dict:
        if "student." in name:
            new_name = name[8:]
            state_dict[new_name] = state_dict.pop(name)
        elif "teacher" in name:
            del state_dict[name]
        elif "distillation_loss" in name or "dist_loss" in name:
            del state_dict[name]
        elif "contrastive_loss.T" in state_dict:
            state_dict["contrastive_loss.loss.T"] = state_dict["contrastive_loss.T"]
            del state_dict["contrastive_loss.T"]

    model.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('OTTER eval script', parents=[get_args_parser()])
    args = parser.parse_args()
    # Eval currently doesn't support distributed data parallel.
    args.distributed = False
    args.teacher_image_arch = ''
    args.teacher_text_arch = ''
    args.use_text_embedding = False
    args.image_aug = False
    args.text_aug = False
    args.distill = False
    args.ema_distill = False
    args.ot_distill = False
    args.label_smoothing = 0.0
    main(args)
