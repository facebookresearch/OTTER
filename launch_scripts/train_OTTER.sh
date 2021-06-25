# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --image_arch resnet50 \
    --text_arch declutr-sci-base \
    --from_pretrained \
    --batch 64 \
    --epochs 10 \
    --dataset cc \
    --data_root <data_root> \
    --num_workers 16 \
    --output_dir <output_path> \
    --lr 3e-3 \
    --wd 0.0 \
    --alpha 0.5 \
    --distill \
    --ema_distill \
    --ot_distill \
    --sinkhorn_iter 5 \
    --sinkhorn_lambda 0.15 \
    --vv_coef 1.0 \
    --tt_coef 1.0 \
    --global_ot \
    --remove_diag \
    --world_size 8