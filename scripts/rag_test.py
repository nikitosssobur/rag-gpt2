from rag.generator import Generator
from rag.retriever import Retriever
from rag.embeddings import EmbeddingModel
from rag.vector_db import FaissVectorDB
from model_loader import load_gpt2
from rag.paths import VECTOR_DB_PATH


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



if __name__ == '__main__':
    vector_db = FaissVectorDB.load(VECTOR_DB_PATH)
    embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    retriever = Retriever(embedding_model=embed_model, vector_db=vector_db)
    tokenizer, lang_model = load_gpt2()
    generator = Generator(lang_model, tokenizer)

    user_request = 'What is April?'
    contexts = retriever.retrieve(user_request)
    response = generator.generate(user_request, contexts)

    print(response)
