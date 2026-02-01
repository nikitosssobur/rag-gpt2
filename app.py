from fastapi import FastAPI
from pydantic import BaseModel
from rag.generator import Generator
from rag.retriever import Retriever
from rag.vector_db import FaissVectorDB
from rag.embeddings import EmbeddingModel
from rag.paths import VECTOR_DB_PATH
from scripts.model_loader import load_gpt2



EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str



app = FastAPI(title="RAG GPT-2 Chatbot")

vector_db = FaissVectorDB.load(VECTOR_DB_PATH)
embed_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
retriever = Retriever(embedding_model=embed_model, vector_db=vector_db)
tokenizer, lang_model = load_gpt2()
generator = Generator(lang_model, tokenizer)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = retriever.retrieve(req.question)
    answer = generator.generate(req.question, context)
    return {"answer": answer}