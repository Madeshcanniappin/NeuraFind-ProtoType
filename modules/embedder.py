import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import read_file, extract_metadata

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_STORE = "vector_store.pkl"

def build_or_load_index(data_folder):
    if os.path.exists(VECTOR_STORE):
        with open(VECTOR_STORE, "rb") as f:
            index, file_metadata = pickle.load(f)
        return index, file_metadata

    print("⚡ Building new vector index...")
    file_texts, metadata_list = [], []
    for file in os.listdir(data_folder):
        filepath = os.path.join(data_folder, file)
        text = read_file(filepath)
        metadata = extract_metadata(filepath)
        metadata["file"] = file
        file_texts.append(text)
        metadata_list.append(metadata)

    embeddings = MODEL.encode(file_texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(VECTOR_STORE, "wb") as f:
        pickle.dump((index, metadata_list), f)

    return index, metadata_list
