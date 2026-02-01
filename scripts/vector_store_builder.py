from rag.vector_db import FaissVectorDB
from rag.embeddings import EmbeddingModel
from wiki_data_preprocessing import create_chuncks_from_folder
from rag.paths import SAMPLE_DATA_PATH, VECTOR_DB_PATH



EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


if __name__ == '__main__':
    vector_db = FaissVectorDB(dim=384)
    emb_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    full_db_text_blocks = create_chuncks_from_folder(folder_path=SAMPLE_DATA_PATH)
    full_db_embeds = emb_model.embed_batch(full_db_text_blocks)
    vector_db.add(full_db_embeds, full_db_text_blocks)
    vector_db.save(VECTOR_DB_PATH)

    print("Vector DB created and saved successfully")