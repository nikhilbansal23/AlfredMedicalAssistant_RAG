from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os
logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading existing vector store from {DB_FAISS_PATH}")
            return FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
        else:
            logger.warning("No existing vector store found, returning None")
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise CustomException(f"Failed to load vector store: {e}")

#create a new vector store
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save vector store")
        logger.info(f"Generating your new vector store with {len(text_chunks)} text chunks to {DB_FAISS_PATH}")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info(f"Saving vector store to {DB_FAISS_PATH}")
        db.save_local(DB_FAISS_PATH)
        logger.info("Vector store saved successfully")
        return db
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
        raise CustomException(f"Failed to create vector store: {e}")
        
