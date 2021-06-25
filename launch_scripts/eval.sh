# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

python eval.py \
  --image_arch resnet50 \
  --text_arch declutr-sci-base \
  --dataset open-images,tencent \
  --data_root <data_root> \
  --embedding_dim 768 \
  -c <path_to_checkpoint>