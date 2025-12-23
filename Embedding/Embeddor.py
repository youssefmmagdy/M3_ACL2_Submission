import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Paths for artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "Ecommerce_KG_Optimized_translated.csv")

# Model 1: MiniLM
EMBEDDINGS_PATH_M1 = os.path.join(ARTIFACTS_DIR, "embeddings_minilm.npy")
INDEX_PATH_M1 = os.path.join(ARTIFACTS_DIR, "faiss_minilm.index")

# Model 2: MPNET
EMBEDDINGS_PATH_M2 = os.path.join(ARTIFACTS_DIR, "embeddings_mpnet.npy")
INDEX_PATH_M2 = os.path.join(ARTIFACTS_DIR, "faiss_mpnet.index")

# Shared chunks metadata
CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.pkl")

# Check if artifacts exist
def _check_artifacts():
    required = [CHUNKS_PATH, EMBEDDINGS_PATH_M1, INDEX_PATH_M1, EMBEDDINGS_PATH_M2, INDEX_PATH_M2]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing embedding artifacts. Please run Embedding/Embeddor.ipynb first.\n"
            f"Missing: {missing}"
        )

# Lazy loading - only load when first needed
_loaded = False
_df = None
_all_chunks = None
_chunk_to_row = None
_embedder_m1 = None
_embedder_m2 = None
_index_m1 = None
_index_m2 = None


def _load_artifacts():
    global _loaded, _df, _all_chunks, _chunk_to_row
    global _embedder_m1, _embedder_m2, _index_m1, _index_m2
    
    if _loaded:
        return
    
    _check_artifacts()
    
    # Load CSV data
    _df = pd.read_csv(CSV_PATH)
    
    # Load chunks metadata
    with open(CHUNKS_PATH, "rb") as f:
        saved_data = pickle.load(f)
        _all_chunks = saved_data["chunks"]
        _chunk_to_row = saved_data["chunk_to_row"]
    
    # Load models
    print("Loading MiniLM model...")
    _embedder_m1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Loading MPNET model...")
    _embedder_m2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Load FAISS indexes
    _index_m1 = faiss.read_index(INDEX_PATH_M1)
    _index_m2 = faiss.read_index(INDEX_PATH_M2)
    
    _loaded = True
    print(f"Loaded {len(_all_chunks)} chunks, MiniLM index: {_index_m1.ntotal}, MPNET index: {_index_m2.ntotal}")


def _build_results(distances, indices, model_name):
    """Build structured results from FAISS search"""
    results = []
    seen_rows = set()
    
    for i, idx in enumerate(indices[0]):
        row_idx = _chunk_to_row[idx]
        if row_idx in seen_rows:
            continue
        seen_rows.add(row_idx)
        
        row = _df.iloc[row_idx]
        results.append({
            "source": f"embedding_{model_name}",
            "model": model_name,
            "product_id": str(row.get('product_id', 'N/A')),
            "category": str(row.get('product_category_name', 'N/A')),
            "city": str(row.get('customer_city', 'N/A')),
            "state": str(row.get('customer_state', 'N/A')),
            "order_status": str(row.get('order_status', 'N/A')),
            "price": float(row.get('price', 0)) if pd.notna(row.get('price')) else 0,
            "freight_value": float(row.get('freight_value', 0)) if pd.notna(row.get('freight_value')) else 0,
            "description_length": float(row.get('product_description_lenght', 0)) if pd.notna(row.get('product_description_lenght')) else 0,
            "photos_qty": float(row.get('product_photos_qty', 0)) if pd.notna(row.get('product_photos_qty')) else 0,
            "delivery_delay_days": int(row.get('delivery_delay_days', 0)) if pd.notna(row.get('delivery_delay_days')) else 0,
            "review_score": float(row.get('review_score', 0)) if pd.notna(row.get('review_score')) else 0,
            "review_comment": str(row.get('review_comment_message', 'N/A')),
            "sentiment": str(row.get('sentiment_group', 'N/A')),
            "similarity_distance": float(distances[0][i]),
        })
    return results


def get_embedded_records_minilm(query, k=3):
    """Retrieve using MiniLM-L6-v2 (faster, 384D)"""
    _load_artifacts()
    
    query_emb = _embedder_m1.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index_m1.search(query_emb, k)
    
    return _build_results(distances, indices, "MiniLM")


def get_embedded_records_mpnet(query, k=3):
    """Retrieve using MPNET-base-v2 (more accurate, 768D)"""
    _load_artifacts()
    
    query_emb = _embedder_m2.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index_m2.search(query_emb, k)
    
    return _build_results(distances, indices, "MPNET")


def get_embedded_records(query, k=3):
    """Retrieve using both models (for comparison)"""
    return {
        "minilm": get_embedded_records_minilm(query, k),
        "mpnet": get_embedded_records_mpnet(query, k)
    }
