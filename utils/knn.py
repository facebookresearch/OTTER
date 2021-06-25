import faiss


class FaissKNN:
    """
    Fast KNN implementation based on FaissGPU.
    """
    def __init__(self, n_neighbors=1, dim=768):
        self.ngpus = faiss.get_num_gpus()

        cpu_index = faiss.IndexFlatL2(dim)
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
        D, I = self.gpu_index.search(embedding, n_neighbors)
        return D, self.embedding_labels[I]

    def predict(self, embedding):
        D, I = self.gpu_index.search(embedding, 1)
        return self.embedding_labels[I].flatten()
