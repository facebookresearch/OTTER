# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def save_model(args, epoch, model, optimizer, lr_scheduler, is_best):
    if args.teacher_image_arch and args.teacher_text_arch:
        model_to_save = model.module.student
    else:
        model_to_save = model.module

    save_dict = {'epoch': epoch + 1,
                 'state_dict': model_to_save.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lr_scheduler': lr_scheduler.state_dict(),
                 'args': vars(args)}

    save_checkpoint(save_dict, checkpoint=args.output_dir, filename="last.pth.tar")

    if is_best:
        save_checkpoint(save_dict, checkpoint=args.output_dir, filename="best.pth.tar")
