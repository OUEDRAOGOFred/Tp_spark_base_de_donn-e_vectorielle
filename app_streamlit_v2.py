"""
Interface Streamlit pour la recherche sémantique médicale
Étape 4 : Interface utilisateur complète avec métriques et visualisations
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import sys
from datetime import datetime
import json

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration de la page
st.set_page_config(
    page_title="Medical Semantic Search - AI Powered",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé moderne
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Fonts Google */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Cards modernes */
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid #d1fae5;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin: 1.2rem 0;
        border-left: 4px solid #10b981;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
        transform: translateX(4px);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0fdfa;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
    }
    
    /* Inputs */
    .stTextInput input {
        border-radius: 12px;
        border: 2px solid #d1fae5;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f0fdfa 0%, #d1fae5 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES RESSOURCES
# ============================================================================

@st.cache_resource
def load_resources():
    """Charger toutes les ressources nécessaires"""
    resources = {
        'index': None,
        'corpus': None,
        'embeddings': None,
        'model': None,
        'cross_encoder': None
    }
    
    try:
        # Index FAISS
        if os.path.exists('medical_faiss.index'):
            resources['index'] = faiss.read_index('medical_faiss.index')
            st.sidebar.success(f"Index : {resources['index'].ntotal} vecteurs")
        
        # Corpus
        if os.path.exists('docs_medical.csv'):
            resources['corpus'] = pd.read_csv('docs_medical.csv')
            # Nettoyer les espaces blancs au début et à la fin des réponses
            if 'Answer' in resources['corpus'].columns:
                resources['corpus']['Answer'] = resources['corpus']['Answer'].astype(str).str.strip()
            st.sidebar.success(f"Corpus : {len(resources['corpus'])} docs")
        
        # Embeddings
        if os.path.exists('embeddings_medical.npy'):
            resources['embeddings'] = np.load('embeddings_medical.npy')
            st.sidebar.success(f"Encastrements : {resources['embeddings'].shape}")
        
        # Modèle Sentence-Transformer
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            resources['model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            resources['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            st.sidebar.success("Modèles IA chargés")
        except Exception as e:
            st.sidebar.warning(f"Modèles non chargés: {str(e)}")
        
        return resources
        
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return resources

# ============================================================================
# FONCTIONS DE RECHERCHE
# ============================================================================

def semantic_search(query, resources, top_k=10, use_reranking=True, filters=None):
    """Recherche sémantique avec métriques"""
    start_time = time.time()
    
    model = resources['model']
    index = resources['index']
    corpus = resources['corpus']
    cross_encoder = resources['cross_encoder']
    
    if model is None or index is None or corpus is None:
        st.error("Ressources non disponibles")
        return None, None
    
    try:
        # Encoder la requête
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Recherche FAISS
        search_k = 50 if use_reranking and cross_encoder else top_k
        scores, indices = index.search(query_embedding.astype('float32'), search_k)
        
        # Récupérer les résultats
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            
            doc = corpus.iloc[idx]
            
            # Appliquer les filtres
            if filters:
                if filters.get('source') and doc['source_file'] != filters['source']:
                    continue
                if filters.get('category') and doc['medical_category'] != filters['category']:
                    continue
            
            results.append({
                'doc_id': int(doc['doc_id']),
                'question': doc['Question'],
                'answer': doc['Answer'],
                'source': doc['source_file'],
                'category': doc['medical_category'],
                'complexity': doc['complexity'],
                'score': float(score),
                'cross_score': None
            })
        
        # Re-ranking
        if use_reranking and cross_encoder and len(results) > 0:
            pairs = [[query, r['answer']] for r in results]
            cross_scores = cross_encoder.predict(pairs)
            
            for i, score in enumerate(cross_scores):
                results[i]['cross_score'] = float(score)
            
            results = sorted(results, key=lambda x: x['cross_score'], reverse=True)
        
        results = results[:top_k]
        
        # Calculer les métriques
        latency = time.time() - start_time
        
        metrics = {
            'latency': latency,
            'results_count': len(results),
            'used_reranking': use_reranking and cross_encoder is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return results, metrics
        
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        return None, None

def calculate_metrics(results, ground_truth_idx=None):
    """Calculer Recall@10, MRR@10, etc."""
    metrics = {}
    
    if results:
        # Latence déjà calculée
        # Recall@10 (nécessite ground truth)
        if ground_truth_idx is not None:
            doc_ids = [r['doc_id'] for r in results[:10]]
            metrics['recall@10'] = 1.0 if ground_truth_idx in doc_ids else 0.0
            
            # MRR@10
            if ground_truth_idx in doc_ids:
                rank = doc_ids.index(ground_truth_idx) + 1
                metrics['mrr@10'] = 1.0 / rank
            else:
                metrics['mrr@10'] = 0.0
        
        # Moyenne des scores
        metrics['avg_score'] = np.mean([r['score'] for r in results])
        
        if results[0].get('cross_score'):
            metrics['avg_cross_score'] = np.mean([r['cross_score'] for r in results if r['cross_score']])
    
    return metrics

# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_embeddings_umap(embeddings, corpus, query_embedding=None, results_indices=None):
    """Visualiser les embeddings avec UMAP"""
    try:
        from umap import UMAP
        
        with st.spinner("📊 Génération de la visualisation UMAP..."):
            # Réduire la dimension pour la visualisation
            n_samples = min(1000, len(embeddings))
            sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embedding_2d = reducer.fit_transform(sample_embeddings)
            
            # Préparer les données
            df_viz = pd.DataFrame({
                'x': embedding_2d[:, 0],
                'y': embedding_2d[:, 1],
                'category': [corpus.iloc[i]['medical_category'] for i in sample_indices],
                'question': [corpus.iloc[i]['Question'][:50] + '...' for i in sample_indices]
            })
            
            # Créer le graphique
            fig = px.scatter(
                df_viz,
                x='x', y='y',
                color='category',
                hover_data=['question'],
                title="Visualisation UMAP des Embeddings (échantillon)",
                width=800, height=600
            )
            
            # Ajouter la requête si présente
            if query_embedding is not None:
                query_2d = reducer.transform(query_embedding)
                fig.add_scatter(
                    x=[query_2d[0, 0]], y=[query_2d[0, 1]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Query'
                )
            
            return fig
    except ImportError:
        st.warning("⚠️ UMAP non installé. Installez avec: pip install umap-learn")
        return None
    except Exception as e:
        st.warning(f"⚠️ Erreur UMAP: {str(e)}")
        return None

def plot_score_distribution(results):
    """Distribution des scores"""
    if not results:
        return None
    
    scores = [r['score'] for r in results]
    cross_scores = [r.get('cross_score', 0) for r in results if r.get('cross_score')]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(1, len(scores) + 1)),
        y=scores,
        name='FAISS Score',
        marker_color='lightblue'
    ))
    
    if cross_scores:
        fig.add_trace(go.Bar(
            x=list(range(1, len(cross_scores) + 1)),
            y=cross_scores,
            name='Cross-Encoder Score',
            marker_color='lightcoral'
        ))
    
    fig.update_layout(
        title="Distribution des Scores par Rang",
        xaxis_title="Rang",
        yaxis_title="Score",
        height=400
    )
    
    return fig

def plot_category_distribution(corpus):
    """Distribution des catégories"""
    category_counts = corpus['medical_category'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Distribution des Catégories Médicales",
        height=400
    )
    
    return fig

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    # En-tête moderne simple
    st.markdown('''
    <div class="main-header animate-fade-in">
        <h1 style="margin: 0; color: white; font-size: 2.8rem; font-weight: 700;">
            <i class="fas fa-hospital" style="margin-right: 15px;"></i>Medical Semantic Search
        </h1>
        <p style="margin: 0.8rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1.2rem; font-weight: 300;">
            Recherche Médicale Intelligente Propulsée par l'IA
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    st.sidebar.markdown('''
        <h2 style="margin: 0 0 20px 0; font-size: 1.4rem; font-weight: 700; color: #10b981;">
            <i class="fas fa-cog" style="margin-right: 8px;"></i>Configuration
        </h2>
    ''', unsafe_allow_html=True)
    
    # Charger les ressources
    resources = load_resources()
    
    # Options de recherche
    st.sidebar.markdown('''
        <h3 style="margin: 20px 0 10px 0; font-size: 1rem; font-weight: 600; color: #059669;">
            <i class="fas fa-search" style="margin-right: 8px;"></i>Options de Recherche
        </h3>
    ''', unsafe_allow_html=True)
    top_k = st.sidebar.slider("Nombre de résultats", 5, 50, 10)
    use_reranking = st.sidebar.checkbox("Utiliser le re-ranking", value=True)
    
    # Filtres
    st.sidebar.markdown('''
        <h3 style="margin: 20px 0 10px 0; font-size: 1rem; font-weight: 600; color: #059669;">
            <i class="fas fa-filter" style="margin-right: 8px;"></i>Filtres
        </h3>
    ''', unsafe_allow_html=True)
    filter_source = st.sidebar.selectbox(
        "Source",
        options=["Toutes"] + (sorted(resources['corpus']['source_file'].unique().tolist()) if resources['corpus'] is not None else [])
    )
    
    filter_category = st.sidebar.selectbox(
        "Catégorie",
        options=["Toutes"] + (sorted(resources['corpus']['medical_category'].unique().tolist()) if resources['corpus'] is not None else [])
    )
    
    # Mode d'affichage
    st.sidebar.markdown('''
        <h3 style="margin: 20px 0 10px 0; font-size: 1rem; font-weight: 600; color: #059669;">
            <i class="fas fa-chart-bar" style="margin-right: 8px;"></i>Visualisations
        </h3>
    ''', unsafe_allow_html=True)
    show_metrics = st.sidebar.checkbox("Afficher les métriques", value=True)
    show_umap = st.sidebar.checkbox("Visualisation UMAP", value=False)
    show_distribution = st.sidebar.checkbox("Distribution des scores", value=True)
    
    # Statistiques globales
    if resources['corpus'] is not None:
        st.sidebar.markdown('''
            <h3 style="margin: 20px 0 10px 0; font-size: 1rem; font-weight: 600; color: #059669;">
                <i class="fas fa-chart-line" style="margin-right: 8px;"></i>Statistiques Globales
            </h3>
        ''', unsafe_allow_html=True)
        st.sidebar.metric("Documents", len(resources['corpus']))
        st.sidebar.metric("Sources", resources['corpus']['source_file'].nunique())
        st.sidebar.metric("Catégories", resources['corpus']['medical_category'].nunique())
    
    # Tabs principales avec SVG
    tab1, tab2, tab3, tab4 = st.tabs([
        f"  Recherche",
        f"  Statistiques", 
        f"  Corpus",
        f"  À propos"
    ])
    
    # ========================================================================
    # TAB 1: RECHERCHE
    # ========================================================================
    with tab1:
        st.markdown('''
            <h2 style="margin: 0 0 24px 0; font-size: 2rem; font-weight: 700; color: #10b981;">
                <i class="fas fa-search" style="margin-right: 12px;"></i>Recherche Sémantique
            </h2>
        ''', unsafe_allow_html=True)
        
        # Initialiser la variable pour les exemples
        if 'example_query' not in st.session_state:
            st.session_state.example_query = ""
        
        # Barre de recherche avec style moderne
        st.markdown("""
            <style>
                .search-container {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    margin-bottom: 1.5rem;
                }
            </style>
        """, unsafe_allow_html=True)
        
        query = st.text_input(
            "Posez votre question médicale",
            placeholder="Ex: What are the symptoms of diabetes? ou Qu'est-ce que le diabète?",
            value=st.session_state.example_query,
            key="search_query",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_button = st.button("Rechercher", type="primary", use_container_width=True)
        
        with col2:
            if st.button("Effacer", use_container_width=True):
                st.rerun()
        
        # Exemples de requêtes avec design moderne
        st.markdown("""
            <div style="margin: 2rem 0 1rem 0;">
                <p style="font-size: 0.9rem; color: #64748b; font-weight: 600; margin-bottom: 0.8rem;">
                    EXEMPLES DE REQUÊTES
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("• Symptômes du diabète", use_container_width=True):
                st.session_state.example_query = "What are the symptoms of diabetes?"
                st.rerun()
        
        with col2:
            if st.button("• Traitement du cancer", use_container_width=True):
                st.session_state.example_query = "What are the treatment options for cancer?"
                st.rerun()
        
        with col3:
            if st.button("• Maladie cardiaque", use_container_width=True):
                st.session_state.example_query = "How to prevent heart disease?"
                st.rerun()
        
        # Effectuer la recherche
        if query and (search_button or query != ""):
            # Préparer les filtres
            filters = {}
            if filter_source != "Toutes":
                filters['source'] = filter_source
            if filter_category != "Toutes":
                filters['category'] = filter_category
            
            # Recherche
            results, search_metrics = semantic_search(
                query, resources, top_k, use_reranking, filters
            )
            
            if results:
                # Afficher les métriques
                if show_metrics and search_metrics:
                    st.markdown("""
                        <h3 style="margin: 1.5rem 0 1rem 0; font-size: 1.4rem; font-weight: 700; color: #10b981;">
                            <i class="fas fa-tachometer-alt" style="margin-right: 10px;"></i>Métriques de Performance
                        </h3>
                    """, unsafe_allow_html=True)
                    # Métriques de recherche avec design moderne
                    st.markdown("""
                        <div style="background: linear-gradient(135deg, #10b98115 0%, #059669 15 100%); 
                                    padding: 1.5rem; border-radius: 16px; margin: 1.5rem 0;">
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Latence", f"{search_metrics['latency']*1000:.1f} ms")
                    
                    with col2:
                        st.metric("Résultats", search_metrics['results_count'])
                    
                    with col3:
                        st.metric("Re-ranking", "Oui" if search_metrics['used_reranking'] else "Non")
                    
                    with col4:
                        avg_score = np.mean([r['score'] for r in results])
                        st.metric("Score Moyen", f"{avg_score:.3f}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Distribution des scores
                if show_distribution:
                    fig_scores = plot_score_distribution(results)
                    if fig_scores:
                        st.plotly_chart(fig_scores, use_container_width=True)
                
                # Visualisation UMAP
                if show_umap and resources['embeddings'] is not None:
                    query_emb = resources['model'].encode([query], normalize_embeddings=True)
                    result_indices = [r['doc_id'] for r in results]
                    
                    fig_umap = plot_embeddings_umap(
                        resources['embeddings'],
                        resources['corpus'],
                        query_emb,
                        result_indices
                    )
                    
                    if fig_umap:
                        st.plotly_chart(fig_umap, use_container_width=True)
                
                # Afficher les résultats
                # Résultats avec design moderne
                st.markdown(f"""
                    <h3 style="margin: 2rem 0 1.5rem 0; font-size: 1.6rem; font-weight: 700; color: #10b981;">
                        <i class="fas fa-book" style="margin-right: 10px;"></i>Résultats ({len(results)})
                    </h3>
                """, unsafe_allow_html=True)
                
                for i, result in enumerate(results, 1):
                    # Score color gradient
                    score_val = result['score']
                    if score_val > 0.8:
                        border_color = "#10b981"  # vert
                    elif score_val > 0.6:
                        border_color = "#3b82f6"  # bleu
                    elif score_val > 0.4:
                        border_color = "#f59e0b"  # orange
                    else:
                        border_color = "#ef4444"  # rouge
                    
                    with st.expander(f"**{i}. {result['question']}**", expanded=(i <= 3)):
                        st.markdown(f"""
                            <div style="background: white; border-left: 4px solid {border_color}; 
                                        padding: 1rem; border-radius: 0 12px 12px 0; margin-bottom: 1rem;">
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**Catégorie:** `{result['category']}`")
                        
                        with col2:
                            st.markdown(f"**Source:** `{result['source'][:20]}...`")
                        
                        with col3:
                            st.markdown(f"**Complexité:** `{result['complexity']}`")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("**Réponse:**")
                        
                        # Afficher la réponse sans espace blanc initial
                        st.markdown(f'<div style="background: #f8fafc; padding: 1.2rem; border-radius: 12px; margin: 1rem 0; line-height: 1.6; color: #1e293b;">{result["answer"].strip()}</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("FAISS Score", f"{result['score']:.4f}")
                        
                        with col2:
                            if result['cross_score']:
                                st.metric("Cross Score", f"{result['cross_score']:.4f}")
                        
                        with col3:
                            st.metric("Doc ID", result['doc_id'])
                
                # Export des résultats
                st.markdown("""
                    <h3 style="margin: 2rem 0 1rem 0; font-size: 1.3rem; font-weight: 700; color: #10b981;">
                        <i class="fas fa-download" style="margin-right: 10px;"></i>Export
                    </h3>
                """, unsafe_allow_html=True)
                
                results_df = pd.DataFrame(results)
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger les résultats (CSV)",
                    data=csv,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning("⚠️ Aucun résultat trouvé")
    
    # ========================================================================
    # TAB 2: STATISTIQUES
    # ========================================================================
    with tab2:
        st.markdown('''
            <h2 style="margin: 0 0 24px 0; font-size: 2rem; font-weight: 700; color: #10b981;">
                <i class="fas fa-chart-pie" style="margin-right: 12px;"></i>Statistiques du Corpus
            </h2>
        ''', unsafe_allow_html=True)
        
        if resources['corpus'] is not None:
            corpus = resources['corpus']
            
            # Métriques générales avec design moderne
            st.markdown("""
                <div style="background: linear-gradient(135deg, #10b98115 0%, #05966915 100%); 
                            padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", len(corpus))
            
            with col2:
                st.metric("Sources", corpus['source_file'].nunique())
            
            with col3:
                st.metric("Catégories", corpus['medical_category'].nunique())
            
            with col4:
                avg_length = corpus['answer_length'].mean()
                st.metric("Longueur Moyenne", f"{avg_length:.0f} cars")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cat = plot_category_distribution(corpus)
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                complexity_counts = corpus['complexity'].value_counts()
                fig_complexity = go.Figure(data=[go.Bar(
                    x=complexity_counts.index,
                    y=complexity_counts.values,
                    marker_color='lightgreen'
                )])
                fig_complexity.update_layout(
                    title="Distribution par Complexité",
                    xaxis_title="Complexité",
                    yaxis_title="Nombre de Documents",
                    height=400
                )
                st.plotly_chart(fig_complexity, use_container_width=True)
            
            # Distribution des longueurs
            fig_lengths = go.Figure()
            
            fig_lengths.add_trace(go.Histogram(
                x=corpus['answer_length'],
                name='Réponses',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig_lengths.update_layout(
                title="Distribution des Longueurs de Réponses",
                xaxis_title="Longueur (caractères)",
                yaxis_title="Fréquence",
                height=400
            )
            
            st.plotly_chart(fig_lengths, use_container_width=True)
            
            # Tableau par source
            st.subheader("📁 Détails par Source")
            source_stats = corpus.groupby('source_file').agg({
                'doc_id': 'count',
                'answer_length': 'mean',
                'medical_category': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
            }).round(0)
            source_stats.columns = ['Nombre de Documents', 'Longueur Moyenne', 'Catégorie Principale']
            st.dataframe(source_stats, use_container_width=True)
        
        else:
            st.warning("⚠️ Corpus non chargé")
    
    # ========================================================================
    # TAB 3: CORPUS
    # ========================================================================
    with tab3:
        st.markdown('''
            <h2 style="margin: 0 0 24px 0; font-size: 2rem; font-weight: 700; color: #10b981;">
                <i class="fas fa-database" style="margin-right: 12px;"></i>Explorer le Corpus
            </h2>
        ''', unsafe_allow_html=True)
        
        if resources['corpus'] is not None:
            corpus = resources['corpus']
            
            # Filtres
            col1, col2 = st.columns(2)
            
            with col1:
                selected_category = st.selectbox(
                    "Filtrer par catégorie",
                    options=["Toutes"] + sorted(corpus['medical_category'].unique().tolist())
                )
            
            with col2:
                selected_source = st.selectbox(
                    "Filtrer par source",
                    options=["Toutes"] + sorted(corpus['source_file'].unique().tolist())
                )
            
            # Appliquer les filtres
            filtered = corpus.copy()
            
            if selected_category != "Toutes":
                filtered = filtered[filtered['medical_category'] == selected_category]
            
            if selected_source != "Toutes":
                filtered = filtered[filtered['source_file'] == selected_source]
            
            st.write(f"**{len(filtered)} documents**")
            
            # Afficher le tableau
            display_cols = ['doc_id', 'Question', 'Answer', 'source_file', 'medical_category', 'complexity']
            st.dataframe(
                filtered[display_cols].head(100),
                use_container_width=True,
                height=600
            )
        
        else:
            st.warning("⚠️ Corpus non chargé")
    
    # ========================================================================
    # TAB 4: À PROPOS
    # ========================================================================
    with tab4:
        st.markdown('<h2 style="color: #10b981; margin-bottom: 2rem;"><i class="fas fa-hospital"></i> Medical Semantic Search Engine</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Ce projet implémente un moteur de recherche sémantique avancé pour des questions-réponses médicales.
        Il utilise des techniques de pointe en NLP et recherche vectorielle pour fournir des résultats pertinents et précis.
        """)
        
        st.markdown('<h3 style="color: #059669; margin-top: 2rem;"><i class="fas fa-tools"></i> Technologies Utilisées</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: #f0fdfa; padding: 1rem; border-radius: 12px; border-left: 3px solid #10b981;">
                <strong style="color: #10b981;"><i class="fas fa-brain"></i> Embeddings</strong><br/>
                <span style="color: #64748b;">Sentence-Transformers<br/>(all-MiniLM-L6-v2)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f0fdfa; padding: 1rem; border-radius: 12px; border-left: 3px solid #0ea5e9;">
                <strong style="color: #0ea5e9;"><i class="fas fa-search"></i> Indexation</strong><br/>
                <span style="color: #64748b;">FAISS<br/>(IndexIVFPQ)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #f0fdfa; padding: 1rem; border-radius: 12px; border-left: 3px solid #059669;">
                <strong style="color: #059669;"><i class="fas fa-sort-amount-down"></i> Re-ranking</strong><br/>
                <span style="color: #64748b;">CrossEncoder<br/>(ms-marco-MiniLM-L-6-v2)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: #f0fdfa; padding: 1rem; border-radius: 12px; border-left: 3px solid #06b6d4;">
                <strong style="color: #06b6d4;"><i class="fas fa-desktop"></i> Interface</strong><br/>
                <span style="color: #64748b;">Streamlit + FastAPI</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: #059669; margin-top: 2rem;"><i class="fas fa-project-diagram"></i> Architecture du Système</h3>', unsafe_allow_html=True)
        
        st.code("""
Query → Encoder → FAISS Search → Re-ranking → Results
              ↓
        Embeddings (384D)
              ↓
     Vector Index (FAISS)
        """, language=None)
        
        st.markdown('<h3 style="color: #059669; margin-top: 2rem;"><i class="fas fa-star"></i> Fonctionnalités</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Recherche sémantique avancée</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Métriques de performance</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Filtres avancés</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Re-ranking avec CrossEncoder</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Visualisation UMAP</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #10b981;"><i class="fas fa-check-circle"></i> Export des résultats</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b;">
            <strong>Projet Big Data & Bases de Données Vectorielles</strong><br/>
            Medical Semantic Search Engine - 2025
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
