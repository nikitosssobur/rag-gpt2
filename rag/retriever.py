



class Retriever:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db



    def retrieve(self, query: str | list, k: int = 3):
        if isinstance(query, str):
            query_vector = self.embedding_model.embed(query)
        elif isinstance(query, (list, tuple)):
            query_vector = self.embedding_model.embed_batch(query)

        return self.vector_db.search(query_vector, k)