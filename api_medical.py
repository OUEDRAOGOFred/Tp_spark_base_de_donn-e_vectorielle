
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Patch pour éviter les erreurs d'import de transformers
import transformers.utils.import_utils as import_utils
original_getattr = import_utils._LazyModule.__getattr__

problematic_callbacks = [
    'CodeCarbonCallback', 'WandbCallback', 'CometCallback',
    'MLflowCallback', 'NeptuneCallback', 'TensorBoardCallback', 'AzureMLCallback'
]

def patched_getattr(self, name):
    if name in problematic_callbacks:
        return None
    try:
        return original_getattr(self, name)
    except Exception:
        return None

import_utils._LazyModule.__getattr__ = patched_getattr

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import faiss
import uvicorn

# Initialiser FastAPI
app = FastAPI(
    title="Medical Q&A Search API",
    version="1.0.0",
    description="API de recherche sémantique pour questions médicales"
)

# Variables globales pour les modèles
model = None
cross_encoder = None
index = None
corpus_df = None

# Utiliser lifespan au lieu de on_event (deprecated)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, cross_encoder, index, corpus_df
    print("Loading resources...")
    
    # Charger les modèles avec gestion d'erreur
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from sentence_transformers import SentenceTransformer, CrossEncoder
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
    
    # Charger les données
    index = faiss.read_index('medical_faiss.index')
    corpus_df = pd.read_csv('docs_medical.csv')
    print(f"✅ Resources loaded. Index contains {index.ntotal} documents")
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Medical Q&A Search API",
    version="1.0.0",
    description="API de recherche sémantique pour questions médicales",
    lifespan=lifespan
)

# Modèles Pydantic
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranking: bool = True

class SearchResult(BaseModel):
    rank: int
    doc_id: int
    question: str
    answer: str
    source: str
    score: float
    cross_score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float

# Endpoints
@app.get("/")
def root():
    return {
        "message": "Medical Q&A Search API",
        "version": "1.0.0",
        "endpoints": ["/query", "/docs/{doc_id}", "/stats"]
    }

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Rechercher des documents pertinents"""
    import time
    start_time = time.time()
    
    try:
        # Vérifier si les modèles sont chargés
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please restart the server."
            )
        
        # Encoder la requête
        query_embedding = model.encode([request.query], normalize_embeddings=True)
        
        # Recherche FAISS
        top_k = 50 if request.use_reranking and cross_encoder is not None else request.top_k
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Préparer les résultats
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = corpus_df.iloc[idx]
            results.append({
                'doc_id': int(doc['doc_id']),
                'question': doc['Question'],
                'answer': doc['Answer'],
                'source': doc['source_file'],
                'score': float(score)
            })
        
        # Re-ranking si demandé et disponible
        if request.use_reranking and cross_encoder is not None:
            pairs = [[request.query, r['answer']] for r in results]
            cross_scores = cross_encoder.predict(pairs)
            
            for result, cross_score in zip(results, cross_scores):
                result['cross_score'] = float(cross_score)
            
            results = sorted(results, key=lambda x: x['cross_score'], reverse=True)[:request.top_k]
        else:
            results = results[:request.top_k]
        
        # Ajouter les rangs
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        processing_time = time.time() - start_time
        
        return {
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "processing_time": processing_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/docs/{doc_id}")
def get_document(doc_id: int):
    """Récupérer un document par son ID"""
    try:
        doc = corpus_df[corpus_df['doc_id'] == doc_id]
        if len(doc) == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = doc.iloc[0]
        return {
            "doc_id": int(doc['doc_id']),
            "question": doc['Question'],
            "answer": doc['Answer'],
            "source": doc['source_file'],
            "topic": doc.get('topic', 'N/A')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Statistiques du corpus"""
    return {
        "total_documents": len(corpus_df),
        "index_size": index.ntotal,
        "sources": corpus_df['source_file'].value_counts().to_dict(),
        "model_loaded": model is not None,
        "cross_encoder_loaded": cross_encoder is not None,
        "model": "all-MiniLM-L6-v2" if model else "Not loaded",
        "cross_encoder": "ms-marco-MiniLM-L-6-v2" if cross_encoder else "Not loaded"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
