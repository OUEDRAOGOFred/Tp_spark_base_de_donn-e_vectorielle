# Guide de Démarrage Rapide

## 🚀 Installation et Lancement en 5 Minutes

### Étape 1: Installer les dépendances

```powershell
pip install -r requirements.txt
```

### Étape 2: Exécuter le pipeline complet

```powershell
.\run_all.bat
```

Ou exécutez manuellement:

```powershell
# 1. Préparer le corpus
python prepare_corpus.py

# 2. Construire l'index
python build_index.py

# 3. Évaluer (optionnel)
python evaluate_search.py
```

### Étape 3: Lancer l'application

**Option A - Interface Streamlit (Recommandé)**
```powershell
.\start_app.bat
```

**Option B - API FastAPI**
```powershell
.\start_api.bat
```

**Option C - Les deux en parallèle**
Ouvrez deux terminaux et lancez les deux scripts.

## 📊 Fichiers Générés

Après exécution du pipeline:

```
✅ docs_medical.csv          # Corpus nettoyé (1500 docs)
✅ embeddings_medical.npy    # Embeddings (2.2 MB)
✅ medical_faiss.index       # Index FAISS (0.8 MB)
✅ index_metadata.pkl        # Métadonnées
✅ evaluation_*.csv          # Résultats évaluation
✅ evaluation_*.png          # Graphiques
```

## 🎯 URLs d'Accès

- **Streamlit UI**: http://localhost:8501
- **API FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc

## ❓ Dépannage

### Problème: ModuleNotFoundError

```powershell
pip install -r requirements.txt --upgrade
```

### Problème: CUDA/GPU

Si vous n'avez pas de GPU, assurez-vous d'utiliser `faiss-cpu`:
```powershell
pip install faiss-cpu --force-reinstall
```

### Problème: Port déjà utilisé

Streamlit:
```powershell
streamlit run app_streamlit_v2.py --server.port 8502
```

FastAPI:
```powershell
uvicorn api_medical_v2:app --port 8001
```

## 📞 Support

Pour toute question, consultez le README.md principal.

Bon développement ! 🚀
