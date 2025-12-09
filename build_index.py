"""
Script de vectorisation et indexation FAISS
Étape 2 : Génération des embeddings et création de l'index
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import pickle
import time

# Configuration
CORPUS_FILE = "docs_medical.csv"
EMBEDDINGS_FILE = "embeddings_medical.npy"
INDEX_FILE = "medical_faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
USE_GPU = False  # Mettre à True si GPU disponible

# Configuration FAISS
USE_IVF = True  # Utiliser IndexIVFPQ pour de meilleures performances
N_CLUSTERS = 100  # Nombre de clusters pour IVF
N_PROBE = 10  # Nombre de clusters à explorer lors de la recherche

def load_corpus():
    """
    Charger le corpus préparé
    """
    print(f"📁 Chargement du corpus depuis {CORPUS_FILE}...")
    
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Le fichier {CORPUS_FILE} n'existe pas. "
            "Exécutez d'abord prepare_corpus.py"
        )
    
    df = pd.read_csv(CORPUS_FILE)
    print(f"✅ Corpus chargé: {len(df)} documents")
    
    return df

def load_model():
    """
    Charger le modèle Sentence-Transformer
    """
    print(f"\n🤖 Chargement du modèle {MODEL_NAME}...")
    
    # Désactiver les warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configurer l'environnement
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    model = SentenceTransformer(MODEL_NAME)
    
    # Mettre sur GPU si disponible et demandé
    if USE_GPU:
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("✅ Modèle chargé sur GPU")
        else:
            print("⚠️  GPU non disponible, utilisation CPU")
    else:
        print("✅ Modèle chargé sur CPU")
    
    return model

def generate_embeddings(df, model):
    """
    Générer les embeddings pour tout le corpus
    """
    print(f"\n🔄 Génération des embeddings...")
    print(f"   Taille du batch: {BATCH_SIZE}")
    
    # Préparer les textes (Question + Answer)
    texts = []
    for _, row in df.iterrows():
        text = f"{row['Question']} {row['Answer']}"
        texts.append(text)
    
    # Générer les embeddings par batch avec barre de progression
    embeddings = []
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Génération"):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)
    
    # Combiner tous les embeddings
    embeddings = np.vstack(embeddings)
    
    elapsed = time.time() - start_time
    print(f"✅ Embeddings générés: {embeddings.shape}")
    print(f"   Temps: {elapsed:.2f}s ({len(texts)/elapsed:.1f} docs/sec)")
    print(f"   Dimension: {embeddings.shape[1]}")
    
    return embeddings

def save_embeddings(embeddings):
    """
    Sauvegarder les embeddings
    """
    print(f"\n💾 Sauvegarde des embeddings dans {EMBEDDINGS_FILE}...")
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"✅ Embeddings sauvegardés ({embeddings.nbytes / 1024 / 1024:.2f} MB)")

def create_faiss_index(embeddings):
    """
    Créer l'index FAISS
    """
    print(f"\n🔨 Création de l'index FAISS...")
    
    dimension = embeddings.shape[0]
    n_vectors = embeddings.shape[0]
    
    print(f"   Nombre de vecteurs: {n_vectors}")
    print(f"   Dimension: {dimension}")
    
    if USE_IVF and n_vectors > N_CLUSTERS * 39:
        # IndexIVFPQ pour de meilleures performances
        print(f"   Type: IndexIVFPQ (clusters={N_CLUSTERS})")
        
        # Quantizer de base
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Index IVF avec Product Quantization
        # M = nombre de sous-vecteurs (doit diviser la dimension)
        # nbits = nombre de bits par sous-vecteur
        m = 8  # Ajuster selon la dimension
        nbits = 8
        
        index = faiss.IndexIVFPQ(
            quantizer,
            embeddings.shape[1],
            N_CLUSTERS,
            m,
            nbits
        )
        
        # Entraîner l'index
        print(f"   Entraînement de l'index...")
        index.train(embeddings.astype('float32'))
        index.nprobe = N_PROBE
        
        print(f"   Ajout des vecteurs...")
        index.add(embeddings.astype('float32'))
        
        print(f"✅ Index IVF créé avec succès")
        
    else:
        # IndexFlatIP pour les petits corpus ou si IVF désactivé
        print(f"   Type: IndexFlatIP (recherche exacte)")
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        print(f"✅ Index Flat créé avec succès")
    
    print(f"   Vecteurs indexés: {index.ntotal}")
    
    return index

def save_index(index):
    """
    Sauvegarder l'index FAISS
    """
    print(f"\n💾 Sauvegarde de l'index dans {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    # Taille du fichier
    file_size = os.path.getsize(INDEX_FILE) / 1024 / 1024
    print(f"✅ Index sauvegardé ({file_size:.2f} MB)")

def test_index(index, embeddings, df):
    """
    Tester l'index avec quelques requêtes
    """
    print(f"\n🧪 Test de l'index...")
    
    # Prendre quelques exemples aléatoires
    test_indices = np.random.choice(len(df), 3, replace=False)
    
    for idx in test_indices:
        query_vec = embeddings[idx:idx+1].astype('float32')
        
        # Recherche
        scores, indices = index.search(query_vec, k=5)
        
        print(f"\n📝 Test avec document {idx}:")
        print(f"   Question: {df.iloc[idx]['Question'][:80]}...")
        print(f"   Top 5 résultats:")
        
        for rank, (score, result_idx) in enumerate(zip(scores[0], indices[0]), 1):
            print(f"      {rank}. [Score: {score:.4f}] {df.iloc[result_idx]['Question'][:60]}...")

def create_metadata():
    """
    Créer un fichier de métadonnées pour l'index
    """
    metadata = {
        'model_name': MODEL_NAME,
        'corpus_file': CORPUS_FILE,
        'embeddings_file': EMBEDDINGS_FILE,
        'index_file': INDEX_FILE,
        'dimension': None,  # Sera rempli plus tard
        'n_vectors': None,
        'index_type': 'IVF' if USE_IVF else 'Flat',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return metadata

def main():
    """
    Pipeline complet de vectorisation et indexation
    """
    print("=" * 60)
    print("🔄 VECTORISATION ET INDEXATION FAISS")
    print("=" * 60)
    
    # Étape 1: Charger le corpus
    df = load_corpus()
    
    # Étape 2: Charger le modèle
    model = load_model()
    
    # Étape 3: Générer les embeddings
    embeddings = generate_embeddings(df, model)
    
    # Étape 4: Sauvegarder les embeddings
    save_embeddings(embeddings)
    
    # Étape 5: Créer l'index FAISS
    index = create_faiss_index(embeddings)
    
    # Étape 6: Sauvegarder l'index
    save_index(index)
    
    # Étape 7: Tester l'index
    test_index(index, embeddings, df)
    
    # Étape 8: Créer les métadonnées
    metadata = create_metadata()
    metadata['dimension'] = embeddings.shape[1]
    metadata['n_vectors'] = embeddings.shape[0]
    
    with open('index_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "=" * 60)
    print("✅ INDEXATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)
    print(f"\n📊 Résumé:")
    print(f"   - Documents indexés: {metadata['n_vectors']}")
    print(f"   - Dimension des embeddings: {metadata['dimension']}")
    print(f"   - Type d'index: {metadata['index_type']}")
    print(f"   - Fichiers créés:")
    print(f"      • {EMBEDDINGS_FILE}")
    print(f"      • {INDEX_FILE}")
    print(f"      • index_metadata.pkl")
    
    return index, embeddings, metadata

if __name__ == "__main__":
    main()
