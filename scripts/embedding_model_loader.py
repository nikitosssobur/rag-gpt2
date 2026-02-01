from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

if __name__ == '__main__':
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)