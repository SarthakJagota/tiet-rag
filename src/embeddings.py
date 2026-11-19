import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"local_files_only": True})
    except Exception:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
