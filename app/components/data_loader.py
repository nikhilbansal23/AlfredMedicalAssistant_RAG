import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import load_vector_store, save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

def process_and_store_documents():
    try:
        logger.info("Starting the document processing pipeline")
        # Load PDF files
        documents = load_pdf_files()
        print(f"[DEBUG] Loaded {len(documents)} documents")
        text_chunks = create_text_chunks(documents)
        print(f"[DEBUG] Created {len(text_chunks)} text chunks")

        save_vector_store(text_chunks)

        logger.info("Vector store has been successfully created and saved")

    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(f"An unexpected error occurred: {e}")
    
if __name__ == "__main__":
    process_and_store_documents()
