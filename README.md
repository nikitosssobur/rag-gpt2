# Basic RAG System (GPT-2 + FAISS)

This repository contains a **minimalistic Retrieval-Augmented Generation (RAG) system**
built with:
- GPT-2 (local large language model, small version)
- SentenceTransformers for embeddings
- FAISS for vector similarity search
- FastAPI for backend API

The project demonstrates a **simple local RAG pipeline**:
text preprocessing → embeddings → vector database → retrieval → generation.

## Requirements

- Python **3.9 – 3.12+**
- Git
- (Optional) NVIDIA GPU + CUDA for faster inference

## Installation

### 1. Clone the repository 
### 2. Create virtual environment
`python -m venv venv`

Activate it:

Windows: `venv\Scripts\activate`

Linux / macOS: `source venv/bin/activate`

### 3. Install dependencies
`pip install --upgrade pip` 

`pip install -r requirements.txt`


## Running and testing
1. Go to the `scripts` folder. 
2. Run `model_loader.py`. 

Calling this file with the code will download the GPT-2 model (via Hugging Face Transformers) 
and its tokenizer to your machine once and cashed locally. 
3. Run `embedding_model_loader.py`. 

Calling this file will download `all-MiniLM-L6-v2` (SentenceTransformers) embedding model. 

All listed models are stored in: `~/.cache/huggingface/`. No repeated downloads are required.

4. To check that the model has actually been downloaded, run `llm_test.py`.

5. Run `vector_store_builder.py`. 

Code in this file will create FAISS based vector db located in `data\vector_db` folder. 

The vector database is built on the basis of a portion of the Wikipedia (SimpleEnglish) dataset located along the path 
`data\sample`. 
The dataset has a folder structure with text files of the following type: wiki_00.txt, wiki_01.txt.

You can download this dataset from the link: https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish. 

You can specify a different path to any other part of the data from the same dataset, 
the main thing is that the folder has the same structure as the folder `data\sample`. 
The vector DB is **not committed** to git. Embeddings are 384-dimensional.

6. Run `rag_test.py` for testing simple RAG system functionality.

7. Run `gpt2_rag_demo.py` from root directory for testing chatbot   
with multiple user requests and for measuring its inference speed. 
Type questions in the terminal and `stop chat` to exit.

8. For testing RAG system using FastAPI Backend:
- Start the server: `uvicorn app:app --reload` 
- Open Swagger UI: `http://127.0.0.1:8000/docs`
- Use the /chat endpoint with JSON: `{"question": "Who is Alan Turing?"}`







