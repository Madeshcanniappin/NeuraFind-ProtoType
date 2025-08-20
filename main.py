import os
from modules.embedder import build_or_load_index
from modules.search import semantic_search, metadata_search
from modules.utils import log_results

print("📂 Indexing files from data/ folder...")
index, file_metadata = build_or_load_index("data")

print("✅ Indexing complete. You can now search your files!\n")

while True:
    query = input("\n🔍 Enter your search query (or type 'exit' to quit): ")

    if query.lower() == "exit":
        print("👋 Exiting NeuraFind...")
        break

    print("\n=== Semantic Search Results ===")
    semantic_results = semantic_search(query, index, file_metadata, top_k=3)
    for res in semantic_results:
        print(f"📄 {res['file']}  (score: {res['score']:.4f})")

    print("\n=== Metadata Search Results ===")
    metadata_results = metadata_search(query, file_metadata)
    for res in metadata_results:
        print(f"📄 {res['file']}  (matched on metadata: {res['match']})")

    log_results(query, semantic_results, metadata_results)
