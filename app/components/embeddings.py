from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def get_embedding_model():
    try:
        logger.info("Initializing our HuggingFace embeddings model")
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("HuggingFace embeddings model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error initializing HuggingFace embeddings model: {e}")
        raise CustomException(f"Failed to initialize embeddings model: {e}")
        
