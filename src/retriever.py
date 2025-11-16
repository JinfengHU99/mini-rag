import faiss
import numpy as np
from typing import List


class Retriever:
    """
    A simple vector retriever using FAISS for similarity search.

    Given embeddings and corresponding texts, it can return top-k most similar texts.
    """

    def __init__(self, embeddings: np.ndarray, texts: List[str], normalize: bool = True):
        """
        Args:
            embeddings: numpy.ndarray of shape (N, D)
            texts: list of N strings
            normalize: bool, whether to normalize vectors to unit length
        """
        assert embeddings.shape[0] == len(texts), "Number of embeddings must match number of texts"
        self.texts = texts
        self.embeddings = embeddings.astype('float32')

        # Normalize embeddings for cosine similarity
        if normalize:
            self.embeddings = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)

        self.dim = self.embeddings.shape[1]

        # Create FAISS inner-product index (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def query(self, q_embedding: np.ndarray, k: int = 3) -> List[str]:
        """
        Query top-k most similar texts.

        Args:
            q_embedding: numpy.ndarray, shape (1, D) or (D,)
            k: number of top results

        Returns:
            List[str]: top-k most similar texts
        """
        q = q_embedding.astype('float32')
        if q.ndim == 1:
            q = np.expand_dims(q, 0)

        # Normalize query vector
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        distances, indices = self.index.search(q, k)
        inds = indices[0].tolist()

        # Return corresponding texts, ignore -1 indices
        res = [self.texts[i] for i in inds if i != -1]
        return res
