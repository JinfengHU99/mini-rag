from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    A wrapper class for generating sentence embeddings using Sentence-Transformers.

    This class provides:
    - encode(): convert text(s) to vector embeddings
    - normalize(): convert embeddings to unit vectors for cosine similarity
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Name of the embedding model
        self.model_name = model_name
        # Load the pre-trained SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size: int = 32):
        """
        Encode text or list of texts into embeddings.

        Args:
            texts: str or list of str, input text(s)
            batch_size: int, number of texts processed per batch

        Returns:
            numpy.ndarray: shape (num_texts, embedding_dim)
        """
        # If a single string is provided, convert it to a list
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings as numpy array
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # If only one text, ensure embeddings is 2D
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)

        return embs

    def normalize(self, embeddings):
        """
        Normalize embeddings to unit vectors (length=1) for cosine similarity.

        Args:
            embeddings: numpy.ndarray of shape (num_texts, embedding_dim)

        Returns:
            numpy.ndarray: normalized embeddings
        """
        # Compute L2 norm of each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Prevent division by zero
        norms[norms == 0] = 1e-9
        # Divide each embedding by its norm
        return embeddings / norms
