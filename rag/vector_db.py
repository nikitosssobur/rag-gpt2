import os

import faiss
import numpy as np
import pickle
import json



class FaissVectorDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts = []


    def add(self, vectors: np.ndarray, texts: list[str]):
        self.index.add(vectors.astype("float32"))
        self.texts.extend(texts)


    def search(self, query_vector: np.ndarray, k: int = 5):
        scores, indices = self.index.search(
            query_vector.reshape(1, -1).astype("float32"), k)
        return [self.texts[i] for i in indices[0]]


    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        faiss.write_index(self.index, f"{path}/vector_db_index.index")
        with open(f"{path}/vector_db_text_chunks.pkl", "wb") as f:
            pickle.dump(self.texts, f)

        with open(f"{path}/dim.json", 'w') as f:
            json.dump({'dim': self.dim}, f)

    @staticmethod
    def load(path: str):
        index = faiss.read_index(f"{path}/vector_db_index.index")
        with open(f"{path}/vector_db_text_chunks.pkl", "rb") as f:
            texts = pickle.load(f)

        with open(f"{path}/dim.json", 'r') as f:
            dim = json.load(f)['dim']

        vector_db = FaissVectorDB(dim=dim)
        vector_db.index = index
        vector_db.texts = texts
        return vector_db