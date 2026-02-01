from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class EmbeddingModel:
    def __init__(self, model_name: str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)


    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)


    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)