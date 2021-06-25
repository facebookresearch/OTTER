# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch

import timm
from transformers import AutoTokenizer, AutoModel
from models.model import ImageModel, TextModel, ImageTextModel, DistillationModel
import utils.utils as utils


def _build_image_model(image_arch, embedding_dim, pretrained=True, lock=False):
    # When loading pretrained weights, num_classes is not passed in to avoid error.
    # The classifier will be reset in ImageModel if dimension of pretrained model
    # doesn't match embedding_dim.
    image_model = timm.create_model(image_arch, pretrained=pretrained)
    if lock:
        for p in image_model.parameters():
            p.requires_grad = False

    image_model = ImageModel(
        image_arch,
        image_model,
        embedding_dim
    )
    return image_model


_TEXT_MODEL_REGISTRY = {
    "declutr-small": ["johngiorgi/declutr-small", 768],
    "declutr-base": ["johngiorgi/declutr-base", 768],
    "declutr-sci-base": ["johngiorgi/declutr-sci-base", 768],
    "sentence-transformer-base": ["sentence-transformers/bert-base-nli-mean-tokens", 768],
    "sentence-transformer-large": ["sentence-transformers/bert-large-nli-mean-tokens", 1024],
}


def _build_text_model(text_arch, max_length, embedding_dim, pretrained=True):
    assert text_arch in _TEXT_MODEL_REGISTRY, f"{text_arch} not in {_TEXT_MODEL_REGISTRY.keys()}"

    huggingface_model_name, feature_dim = _TEXT_MODEL_REGISTRY[text_arch]
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name, use_fast=False)
    text_model = AutoModel.from_pretrained(huggingface_model_name)
    if not pretrained:
        text_model.init_weights()
    text_model = TextModel(
        text_arch,
        text_model,
        tokenizer,
        feature_dim=feature_dim,
        out_dim=embedding_dim,
        max_length=max_length,
    )
    return text_model


def build_image_text_model(
    image_arch,
    text_arch,
    embedding_dim,
    max_token_length=60,
    label_smoothing=0.0,
    pretrain=True,
    lock_image=False
):
    print("Initializing models from pretrained weights: ", pretrain)
    # Build Image Model
    image_model = _build_image_model(image_arch, embedding_dim, pretrain, lock_image)
    print("Image Model Params: ", sum(p.numel() for p in image_model.parameters()))

    if utils.is_dist_avail_and_initialized():
        image_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_model)

    # Build Text Model
    text_model = _build_text_model(text_arch, max_token_length, embedding_dim, pretrain)
    print("Text Model Params: ", sum(p.numel() for p in text_model.parameters()))

    image_text_model = ImageTextModel(image_model, text_model, label_smoothing)
    return image_text_model


def build_model(args):
    model = build_image_text_model(
        args.image_arch,
        args.text_arch,
        args.embedding_dim,
        args.max_token_length,
        args.label_smoothing,
        args.from_pretrained,
        args.lock_image_encoder
    )
    print("Model Params: ", sum(p.numel() for p in model.parameters()))

    # Distillation
    if args.distill:
        if args.ema_distill:
            # Use EMA as teacher.
            teacher_model = copy.deepcopy(model)
        else:
            # Load pretrained fixed teacher model.
            teacher_model = build_image_text_model(
                args.teacher_image_arch,
                args.teacher_text_arch,
                args.embedding_dim,
                args.max_token_length,
                args.label_smoothing,
            )
            if args.teacher_checkpoint:
                state_dict = torch.load(args.teacher_checkpoint)['state_dict']
                teacher_model.load_state_dict(state_dict)

        # Turn off gradient for teacher model.
        for p in teacher_model.parameters():
            p.requires_grad = False
        print("Teacher Model Params: ", sum(p.numel() for p in teacher_model.parameters()))

        model = DistillationModel(
            student=model,
            teacher=teacher_model,
            alpha=args.alpha,
            ema=args.ema_distill,
            ema_decay=args.ema_decay,
            T_t=args.distill_T_t,
            ot_dist=args.ot_distill,
            sinkhorn_lambda=args.sinkhorn_lambda,
            sinkhorn_iter=args.sinkhorn_iter,
            vv_coef=args.vv_coef,
            tt_coef=args.tt_coef,
            global_ot=args.global_ot,
            remove_diag=args.remove_diag,
        )

    return model
