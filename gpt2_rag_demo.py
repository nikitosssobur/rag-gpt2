from rag.generator import Generator
from rag.retriever import Retriever
from rag.embeddings import EmbeddingModel
from rag.vector_db import FaissVectorDB
from scripts.model_loader import load_gpt2
from rag.paths import VECTOR_DB_PATH
import time



EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



if __name__ == '__main__':
    vector_db_load_start_time = time.time()
    vector_db = FaissVectorDB.load(VECTOR_DB_PATH)
    vector_db_load_time = time.time() - vector_db_load_start_time

    print(f"Vector DB loading time: {vector_db_load_time}")

    embedding_model_load_start_time = time.time()
    embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    embedding_model_load_time = time.time() - embedding_model_load_start_time

    print(f"Embedding model loading time: {embedding_model_load_time}")


    retriever = Retriever(embedding_model=embed_model, vector_db=vector_db)


    tok_llm_load_start_time = time.time()
    tokenizer, lang_model = load_gpt2()
    tok_llm_load_time = time.time() - tok_llm_load_start_time

    print(f"GPT-2 language model and tokenizer loading time: {tok_llm_load_time} \n")

    generator = Generator(lang_model, tokenizer)


    user_request = ""

    while user_request.lower() != "stop chat":
        user_request = input("Enter your request: ")

        general_chatbot_inf_start_time = time.time()
        retriever_start_time = time.time()
        contexts = retriever.retrieve(user_request)
        retriever_time = time.time() - retriever_start_time

        response_generation_start_time = time.time()
        response = generator.generate(user_request, contexts)
        response_generation_time = time.time() - response_generation_start_time

        general_chatbot_inf_time = time.time() - general_chatbot_inf_start_time

        print(response)

        print(f"General chatbot inference time (retriever and generator): {general_chatbot_inf_time}")

        print(f"Retriever work time: {retriever_time}")

        print(f"Response generation time: {response_generation_time}")




