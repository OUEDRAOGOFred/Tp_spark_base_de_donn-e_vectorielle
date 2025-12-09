# Guide d'Installation et de Lancement du Projet

## 📋 Prérequis

- Python 3.8 ou supérieur
- Git
- 4 Go de RAM minimum
- 2 Go d'espace disque libre

## 🚀 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/OUEDRAOGOFred/Tp_spark_base_de_donn-e_vectorielle.git
cd Tp_spark_base_de_donn-e_vectorielle
```

### 2. Créer un environnement virtuel (recommandé)
```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📊 Préparation des données

### 4. Préparer le corpus médical
**Option 1 : Avec Pandas (plus rapide pour démarrer)**
```bash
python prepare_corpus.py
```

**Option 2 : Avec Spark (pour gros volumes)**
```bash
python prepare_corpus_spark.py
```

### 5. Construire l'index FAISS
```bash
python build_index.py
```

**Note** : Cette étape peut prendre plusieurs minutes et téléchargera automatiquement le modèle de sentence embeddings (~90 Mo).

## ▶️ Lancer l'Application

### Option A : Interface Streamlit (Recommandée)
```bash
streamlit run app_final.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

**Ou utilisez le script batch (Windows uniquement) :**
```bash
start_app.bat
```

### Option B : API REST FastAPI
```bash
uvicorn api_medical_v2:app --reload
```
L'API sera accessible à : `http://localhost:8000`
Documentation interactive : `http://localhost:8000/docs`

**Ou utilisez le script batch (Windows uniquement) :**
```bash
start_api.bat
```

### Option C : Lancer tout en une commande (Windows)
```bash
run_all.bat
```

## 🧪 Évaluation du système

Pour évaluer les performances du moteur de recherche :
```bash
python evaluate_search.py
```

## 📝 Utilisation

### Interface Streamlit
1. Entrez votre question médicale dans la zone de texte
2. Cliquez sur "Rechercher"
3. Consultez les résultats les plus pertinents avec leurs scores

### API REST
**Exemple de requête :**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of diabetes?", "top_k": 5}'
```

**Exemple avec Python :**
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "What are the symptoms of diabetes?", "top_k": 5}
)
print(response.json())
```

## 🔧 Résolution de problèmes

### Erreur : "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Erreur : "FAISS index not found"
```bash
python build_index.py
```

### Erreur mémoire lors de la construction de l'index
- Réduisez le nombre de documents dans `prepare_corpus.py`
- Augmentez la RAM disponible
- Utilisez un système avec plus de ressources

### Le modèle ne se télécharge pas
Téléchargez manuellement le modèle :
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('model_cache')
```

## 📂 Structure du projet

```
.
├── prepare_corpus.py          # Préparation des données (Pandas)
├── prepare_corpus_spark.py    # Préparation des données (Spark)
├── build_index.py             # Construction de l'index FAISS
├── app_final.py               # Interface Streamlit
├── api_medical_v2.py          # API REST
├── evaluate_search.py         # Évaluation du système
├── requirements.txt           # Dépendances Python
├── BD quest_resp medecine/    # Données sources
└── archive1/                  # Données archivées
```

## 📖 Documentation complémentaire

- [README.md](README.md) - Vue d'ensemble du projet
- [QUICKSTART.md](QUICKSTART.md) - Guide de démarrage rapide
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) - Script de démonstration
- [SPARK_VS_PANDAS.md](SPARK_VS_PANDAS.md) - Comparaison des approches

## ⚙️ Configuration avancée

Pour modifier les paramètres du système, éditez les variables dans les fichiers :
- `build_index.py` : Paramètres de l'index FAISS
- `app_final.py` : Configuration de l'interface
- `api_medical_v2.py` : Configuration de l'API

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le dépôt
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

## 📧 Support

Pour toute question ou problème, ouvrez une issue sur GitHub ou contactez l'équipe du projet.
