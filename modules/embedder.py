import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from modules.utils import extract_text

INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.pkl"
DOCS_PATH = "docs/"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index():
    texts, metadata = [], {}
    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)
        text = extract_text(file_path)
        if text.strip():
            texts.append(text)
            metadata[file] = {"path": file_path}
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    save_index(index, metadata)
    return index, metadata

def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def build_or_load_index():
    # If index exists, load it
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index, metadata = load_index()

        # Scan docs for new files
        updated = False
        for file in os.listdir(DOCS_PATH):
            if file not in metadata:
                print(f"[+] New file detected: {file}")
                file_path = os.path.join(DOCS_PATH, file)
                text = extract_text(file_path)
                if text.strip():
                    embedding = model.encode([text], convert_to_numpy=True)
                    index.add(embedding)
                    metadata[file] = {"path": file_path}
                    updated = True
        
        if updated:
            save_index(index, metadata)
        return index, metadata
    
    # Otherwise, build from scratch
    else:
        return build_index()
