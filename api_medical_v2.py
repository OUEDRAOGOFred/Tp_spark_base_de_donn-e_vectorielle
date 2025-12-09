"""
API REST FastAPI pour la recherche sémantique médicale
Étape 3 : Backend IA avec re-ranking et métriques
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import faiss
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime

warnings.filterwarnings('ignore')

# Variables globales
model = None
cross_encoder = None
index = None
corpus_df = None
embeddings = None
search_history = []

# Modèles Pydantic
class QueryRequest(BaseModel):
    query: str = Field(..., description="Texte de la requête", min_length=3)
    top_k: int = Field(10, description="Nombre de résultats à retourner", ge=1, le=100)
    use_reranking: bool = Field(True, description="Utiliser le re-ranking avec CrossEncoder")
    filters: Optional[Dict[str, str]] = Field(None, description="Filtres optionnels (source, category)")

class SearchResult(BaseModel):
    rank: int
    doc_id: int
    question: str
    answer: str
    source: str
    category: str
    score: float
    cross_score: Optional[float] = None
    
class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    used_reranking: bool
    timestamp: str

class DocumentResponse(BaseModel):
    doc_id: int
    question: str
    answer: str
    source: str
    category: str
    complexity: str
    question_length: int
    answer_length: int

class StatsResponse(BaseModel):
    total_documents: int
    index_type: str
    embedding_dimension: int
    sources: List[str]
    categories: Dict[str, int]
    total_searches: int
    average_latency: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cross_encoder_loaded: bool
    index_loaded: bool
    corpus_loaded: bool
    embeddings_loaded: bool

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, cross_encoder, index, corpus_df, embeddings
    
    print("=" * 60)
    print("🚀 Démarrage de l'API Medical Search...")
    print("=" * 60)
    
    # Charger l'index FAISS
    try:
        print("📁 Chargement de l'index FAISS...")
        index = faiss.read_index('medical_faiss.index')
        print(f"✅ Index chargé: {index.ntotal} vecteurs")
    except Exception as e:
        print(f"❌ Erreur chargement index: {e}")
        index = None
    
    # Charger le corpus
    try:
        print("📁 Chargement du corpus...")
        corpus_df = pd.read_csv('docs_medical.csv')
        print(f"✅ Corpus chargé: {len(corpus_df)} documents")
    except Exception as e:
        print(f"❌ Erreur chargement corpus: {e}")
        corpus_df = None
    
    # Charger les embeddings
    try:
        print("📁 Chargement des embeddings...")
        embeddings = np.load('embeddings_medical.npy')
        print(f"✅ Embeddings chargés: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Erreur chargement embeddings: {e}")
        embeddings = None
    
    # Charger le modèle Sentence-Transformer
    try:
        print("🤖 Chargement du modèle Sentence-Transformer...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"⚠️  Avertissement: {e}")
        print("   L'API fonctionnera en mode dégradé")
        model = None
    
    # Charger le CrossEncoder pour le re-ranking
    try:
        print("🤖 Chargement du CrossEncoder...")
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ CrossEncoder chargé avec succès")
    except Exception as e:
        print(f"⚠️  Avertissement: {e}")
        print("   Re-ranking désactivé")
        cross_encoder = None
    
    print("=" * 60)
    print("✅ API prête à recevoir des requêtes!")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n🛑 Arrêt de l'API...")

# Créer l'application FastAPI
app = FastAPI(
    title="Medical Q&A Semantic Search API",
    version="2.0.0",
    description="API de recherche sémantique pour questions-réponses médicales avec re-ranking",
    lifespan=lifespan
)

# CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
def root():
    """Page d'accueil de l'API"""
    return {
        "name": "Medical Q&A Semantic Search API",
        "version": "2.0.0",
        "description": "Recherche sémantique dans un corpus médical",
        "endpoints": {
            "search": "/query",
            "document": "/docs/{doc_id}",
            "statistics": "/stats",
            "health": "/health"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Vérifier l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if all([model, index, corpus_df, embeddings]) else "degraded",
        model_loaded=model is not None,
        cross_encoder_loaded=cross_encoder is not None,
        index_loaded=index is not None,
        corpus_loaded=corpus_df is not None,
        embeddings_loaded=embeddings is not None
    )

@app.post("/query", response_model=QueryResponse, tags=["Search"])
def search_documents(request: QueryRequest):
    """
    Rechercher des documents pertinents
    
    - **query**: La requête en langage naturel
    - **top_k**: Nombre de résultats souhaités (1-100)
    - **use_reranking**: Activer le re-ranking avec CrossEncoder
    - **filters**: Filtres optionnels par source ou catégorie
    """
    start_time = time.time()
    
    # Vérifications
    if model is None or index is None or corpus_df is None:
        raise HTTPException(
            status_code=503,
            detail="Service non disponible. Modèles non chargés."
        )
    
    try:
        # Encoder la requête
        query_embedding = model.encode(
            [request.query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Recherche FAISS
        # Si re-ranking activé, chercher plus de résultats
        search_k = 50 if request.use_reranking and cross_encoder else request.top_k
        scores, indices = index.search(query_embedding.astype('float32'), search_k)
        
        # Récupérer les documents
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # Pas de résultat trouvé
                continue
                
            doc = corpus_df.iloc[idx]
            
            # Appliquer les filtres si présents
            if request.filters:
                skip = False
                if 'source' in request.filters:
                    if doc['source_file'] != request.filters['source']:
                        skip = True
                if 'category' in request.filters:
                    if doc['medical_category'] != request.filters['category']:
                        skip = True
                if skip:
                    continue
            
            results.append({
                'doc_id': int(doc['doc_id']),
                'question': doc['Question'],
                'answer': doc['Answer'],
                'source': doc['source_file'],
                'category': doc['medical_category'],
                'score': float(score),
                'cross_score': None
            })
        
        # Re-ranking avec CrossEncoder si activé
        if request.use_reranking and cross_encoder and len(results) > 0:
            # Préparer les paires (query, answer)
            pairs = [[request.query, r['answer']] for r in results]
            
            # Calculer les scores de re-ranking
            cross_scores = cross_encoder.predict(pairs)
            
            # Mettre à jour les scores
            for i, score in enumerate(cross_scores):
                results[i]['cross_score'] = float(score)
            
            # Re-trier par cross_score
            results = sorted(results, key=lambda x: x['cross_score'], reverse=True)
        
        # Limiter au top_k demandé
        results = results[:request.top_k]
        
        # Ajouter le rang
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        processing_time = time.time() - start_time
        
        # Sauvegarder dans l'historique
        search_history.append({
            'query': request.query,
            'timestamp': datetime.now().isoformat(),
            'latency': processing_time,
            'results_count': len(results)
        })
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            used_reranking=request.use_reranking and cross_encoder is not None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche: {str(e)}"
        )

@app.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
def get_document(doc_id: int):
    """
    Récupérer un document spécifique par son ID
    """
    if corpus_df is None:
        raise HTTPException(
            status_code=503,
            detail="Corpus non chargé"
        )
    
    try:
        doc = corpus_df[corpus_df['doc_id'] == doc_id]
        
        if len(doc) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} non trouvé"
            )
        
        doc = doc.iloc[0]
        
        return DocumentResponse(
            doc_id=int(doc['doc_id']),
            question=doc['Question'],
            answer=doc['Answer'],
            source=doc['source_file'],
            category=doc['medical_category'],
            complexity=doc['complexity'],
            question_length=int(doc['question_length']),
            answer_length=int(doc['answer_length'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
def get_statistics():
    """
    Obtenir des statistiques sur le corpus et l'utilisation de l'API
    """
    if corpus_df is None or index is None:
        raise HTTPException(
            status_code=503,
            detail="Données non chargées"
        )
    
    try:
        # Calculer la latence moyenne
        avg_latency = 0.0
        if search_history:
            avg_latency = sum(h['latency'] for h in search_history) / len(search_history)
        
        # Distribution par catégorie
        category_dist = corpus_df['medical_category'].value_counts().to_dict()
        
        return StatsResponse(
            total_documents=len(corpus_df),
            index_type="FAISS IVF" if hasattr(index, 'nprobe') else "FAISS Flat",
            embedding_dimension=embeddings.shape[1] if embeddings is not None else 0,
            sources=sorted(corpus_df['source_file'].unique().tolist()),
            categories=category_dist,
            total_searches=len(search_history),
            average_latency=avg_latency
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/sources", tags=["Metadata"])
def get_sources():
    """Obtenir la liste des sources disponibles"""
    if corpus_df is None:
        raise HTTPException(status_code=503, detail="Corpus non chargé")
    
    return {
        "sources": sorted(corpus_df['source_file'].unique().tolist())
    }

@app.get("/categories", tags=["Metadata"])
def get_categories():
    """Obtenir la liste des catégories médicales"""
    if corpus_df is None:
        raise HTTPException(status_code=503, detail="Corpus non chargé")
    
    categories = corpus_df['medical_category'].value_counts().to_dict()
    return {"categories": categories}

@app.get("/history", tags=["Statistics"])
def get_search_history(limit: int = Query(50, ge=1, le=1000)):
    """Obtenir l'historique des recherches"""
    return {
        "total": len(search_history),
        "history": search_history[-limit:]
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
