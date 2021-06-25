# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss


class FaissKNN:
    """
    Fast KNN implementation based on FaissGPU.
    """
    def __init__(self, n_neighbors=1, dim=768, mode='cos'):
        self.ngpus = faiss.get_num_gpus()
        print("Faiss num gpus: ", self.ngpus)
        self.mode = mode
        if mode == 'cos':
            cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        else:
            raise NotImplementedError
        self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        self.k = n_neighbors

    def reset(self):
        self.gpu_index.reset()

    def fit(self, embedding_bank, embedding_labels):
        self.gpu_index.add(embedding_bank)
        self.embedding_labels = embedding_labels

    def kneighbors(self, embedding, n_neighbors=None):
        if not n_neighbors:
            n_neighbors = self.k
        D, Ind = self.gpu_index.search(embedding, n_neighbors)
        return D, self.embedding_labels[Ind]

    def predict(self, embedding):
        D, Ind = self.gpu_index.search(embedding, 1)
        return self.embedding_labels[Ind].flatten()
