import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, index, file_metadata, top_k=3):
    query_vec = MODEL.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(file_metadata):
            results.append({
                "file": file_metadata[idx]["file"],
                "score": 1 / (1 + score)
            })
    return results


def metadata_search(query, file_metadata):
    query_lower = query.lower()
    results = []

    for meta in file_metadata:
        for key, value in meta.items():
            if query_lower in str(value).lower():
                results.append({
                    "file": meta["file"],
                    "match": f"{key}: {value}"
                })
    return results
