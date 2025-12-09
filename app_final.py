import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.graph_objects as go
import time
import pickle
import os
import sys
from types import ModuleType
from sklearn.metrics.pairwise import cosine_similarity

# Configuration CRITIQUE - AVANT TOUS LES IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'  # Désactiver TensorFlow
os.environ['USE_TORCH'] = '1'  # Forcer PyTorch

# Créer callback factice
class DummyCallback:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return DummyCallback()

# Créer modules factices
class DummyModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = type('ModuleSpec', (), {'name': name, 'origin': None, 'loader': None})()
        self.__file__ = f'<dummy {name}>'
        self.__path__ = []
    def __getattr__(self, name):
        return DummyCallback

# Bloquer modules problématiques
for name in ['codecarbon', 'wandb', 'comet_ml', 'mlflow', 'neptune', 'tensorboard', 'azureml', 'dvclive', 'clearml', 'dagshub']:
    if name not in sys.modules:
        sys.modules[name] = DummyModule(name)

# Patcher transformers
try:
    import transformers
    if hasattr(transformers, 'integrations'):
        for callback in ['CodeCarbonCallback', 'WandbCallback', 'CometCallback', 'MLflowCallback', 
                        'NeptuneCallback', 'TensorBoardCallback', 'AzureMLCallback', 'DVCLiveCallback',
                        'ClearMLCallback', 'DagsHubCallback', 'FlyteCallback', 'SageMakerCallback']:
            setattr(transformers.integrations, callback, DummyCallback)
except:
    pass

# Maintenant importer sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False
    SentenceTransformer = None

# Configuration de la page
st.set_page_config(
    page_title="Recherche Médicale Sémantique",
    page_icon="🏥",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Charger les ressources sans les modèles lourds
@st.cache_resource
def load_resources():
    """Charger les données pré-calculées"""
    try:
        # Charger l'index FAISS
        index = faiss.read_index('medical_faiss.index')
        
        # Charger le corpus
        corpus_df = pd.read_csv('docs_medical.csv')
        
        # Charger les embeddings pré-calculés
        embeddings = np.load('embeddings_medical.npy')
        
        st.success("✅ Données chargées avec succès!")
        
        # Charger le modèle avec le patch
        model = None
        has_model = False
        
        if MODEL_AVAILABLE and SentenceTransformer is not None:
            try:
                st.info("🔄 Chargement du modèle Sentence-Transformer...")
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                has_model = True
                st.success("✅ Modèle chargé avec succès!")
            except Exception as e:
                st.warning(f"⚠️ Erreur chargement modèle: {str(e)[:100]}")
                st.info("ℹ️ Utilisation de la recherche hybride")
        else:
            st.info("ℹ️ Recherche hybride activée")
        
        return index, corpus_df, embeddings, model, has_model
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return None, None, None, None, False
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        st.stop()

# Fonction de recherche par similarité
def search_documents(query_text, index, corpus_df, embeddings, model, has_model, top_k=10):
    """Recherche avec FAISS"""
    start_time = time.time()
    
    try:
        if has_model and model is not None:
            # Utiliser le modèle pour encoder la requête
            query_embedding = model.encode([query_text], normalize_embeddings=True)
        else:
            # Approche optimisée: recherche en deux étapes
            st.info("🔍 Recherche sémantique avancée (embeddings pré-calculés)")
            
            # Étape 1: Trouver les documents les plus pertinents par mots-clés
            query_lower = query_text.lower()
            query_words = set(query_lower.split())
            
            # Calculer le score TF-IDF manuel pour chaque document
            doc_scores = []
            for idx, row in corpus_df.iterrows():
                doc_text = (row['Question'] + " " + row['Answer']).lower()
                doc_words = doc_text.split()
                
                # Score basé sur la fréquence et la rareté des termes
                score = 0
                for word in query_words:
                    if word in doc_text:
                        # TF: fréquence du terme dans le document
                        tf = doc_words.count(word) / len(doc_words)
                        # IDF simulé: mots rares valent plus
                        idf = np.log(len(corpus_df) / (1 + sum(word in str(row['Question']).lower() for _, row in corpus_df.iterrows())))
                        score += tf * idf
                
                if score > 0:
                    doc_scores.append((idx, score))
            
            if doc_scores:
                # Trier et prendre les top 30 documents
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                top_docs = doc_scores[:min(30, len(doc_scores))]
                
                # Étape 2: Créer un vecteur de requête optimal
                # Utiliser une combinaison pondérée des meilleurs embeddings
                total_weight = sum(score for _, score in top_docs)
                query_embedding = np.zeros((1, embeddings.shape[1]), dtype='float32')
                
                for idx, score in top_docs:
                    weight = score / total_weight
                    query_embedding += weight * embeddings[idx:idx+1]
                
                # Normaliser pour la recherche cosinus
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                else:
                    # Fallback si aucun match
                    query_embedding = embeddings[0:1]
            else:
                # Si aucun mot ne matche, utiliser une recherche plus large
                # Prendre un échantillon aléatoire pondéré
                sample_size = min(50, len(corpus_df))
                random_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                query_embedding = np.mean(embeddings[random_indices], axis=0, keepdims=True).astype('float32')
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
        
        # Recherche FAISS avec le vecteur optimisé
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            doc = corpus_df.iloc[idx]
            results.append({
                'rank': rank,
                'doc_id': int(doc['doc_id']),
                'question': doc['Question'],
                'answer': doc['Answer'],
                'source': doc['source_file'],
                'score': float(score)
            })
        
        elapsed = time.time() - start_time
        return results, elapsed
        
    except Exception as e:
        st.error(f"Erreur de recherche: {str(e)}")
        return [], 0

# Interface principale
def main():
    # Titre
    st.markdown('<h1 class="main-header">🏥 Recherche Médicale Sémantique</h1>', unsafe_allow_html=True)
    st.markdown("### Système de recherche intelligent pour questions médicales")
    
    # Charger les ressources
    with st.spinner("Chargement des données..."):
        index, corpus_df, embeddings, model, has_model = load_resources()
    
    # Barre latérale
    with st.sidebar:
        st.header("⚙️ Paramètres")
        top_k = st.slider("Nombre de résultats", 1, 20, 10)
        
        st.markdown("---")
        st.header("📊 Statistiques")
        st.metric("Documents totaux", len(corpus_df))
        st.metric("Vecteurs indexés", index.ntotal)
        st.metric("Modèle actif", "✅ Oui" if has_model else "❌ Non")
        
        st.markdown("---")
        st.header("📚 Sources")
        source_counts = corpus_df['source_file'].value_counts()
        for source, count in source_counts.head(5).items():
            st.text(f"{source[:20]}...: {count}")
    
    # Zone de recherche
    st.markdown("### 🔍 Entrez votre question médicale")
    query = st.text_input(
        "",
        placeholder="Ex: What are the symptoms of diabetes?",
        label_visibility="collapsed",
        key="query_input"
    )
    
    # Exemples de requêtes
    st.markdown("**Exemples de questions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🩺 Symptoms of cancer"):
            st.session_state.query_input = "What are the symptoms of cancer?"
            query = "What are the symptoms of cancer?"
    with col2:
        if st.button("💊 Treatment for diabetes"):
            st.session_state.query_input = "What is the treatment for diabetes?"
            query = "What is the treatment for diabetes?"
    with col3:
        if st.button("🧠 Neurological disorders"):
            st.session_state.query_input = "What are common neurological disorders?"
            query = "What are common neurological disorders?"
    
    # Recherche
    if query:
        # Afficher clairement la question posée
        st.markdown("---")
        st.markdown("### 💬 Votre question")
        st.info(query)
        
        with st.spinner("Recherche en cours..."):
            results, elapsed = search_documents(
                query, index, corpus_df, embeddings, model, has_model, top_k
            )
        
        if results:
            # Afficher les métriques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Temps", f"{elapsed:.3f}s")
            with col2:
                st.metric("📄 Résultats", len(results))
            with col3:
                avg_score = np.mean([r['score'] for r in results])
                st.metric("🎯 Score moyen", f"{avg_score:.3f}")
            
            st.markdown("---")
            
            # Afficher les résultats
            st.markdown("### 📋 Documents similaires trouvés")
            for result in results:
                with st.expander(
                    f"#{result['rank']} - Score: {result['score']:.3f} - {result['question'][:70]}...",
                    expanded=(result['rank'] == 1)
                ):
                    st.markdown(f"**Question du document:** {result['question']}")
                    st.markdown(f"**Réponse:** {result['answer']}")
                    st.caption(f"📁 Source: {result['source']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"📁 Source: {result['source']}")
                    with col2:
                        st.caption(f"🔢 Doc ID: {result['doc_id']}")
            
            # Visualisation
            st.markdown("---")
            st.markdown("### 📊 Distribution des scores")
            
            scores_df = pd.DataFrame(results)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores_df['rank'],
                y=scores_df['score'],
                marker_color='lightblue',
                text=scores_df['score'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="Scores de pertinence par rang",
                xaxis_title="Rang",
                yaxis_title="Score",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun résultat trouvé")

if __name__ == "__main__":
    main()
