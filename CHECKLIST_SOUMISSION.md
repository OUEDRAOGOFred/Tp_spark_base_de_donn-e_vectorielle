# ✅ Checklist avant de soumettre le projet

## 📋 Vérifications essentielles

### 1. ✅ Fichiers présents
- [x] Code source (`.py`)
- [x] Documentation (`README.md`, `INSTALLATION.md`)
- [x] Requirements (`requirements.txt`)
- [x] Données (fichiers CSV dans `BD quest_resp medecine/`)
- [x] Scripts de lancement (`.bat`)

### 2. ⚠️ Fichiers exclus (trop volumineux)
Les fichiers suivants sont exclus du dépôt GitHub (voir `.gitignore`) :
- ❌ `model_cache/` (~90 MB) - Sera téléchargé automatiquement
- ❌ `embeddings_medical.npy` - Sera généré avec `build_index.py`
- ❌ `medical_faiss.index` - Sera généré avec `build_index.py`
- ❌ Certains CSV très volumineux

**👉 C'est normal !** Ces fichiers seront recréés automatiquement lors de l'installation.

### 3. 🐍 Versions Python compatibles
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ⚠️ Python 3.12 (peut nécessiter des ajustements mineurs)

### 4. 💻 Systèmes d'exploitation testés
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS (avec quelques adaptations)

## 🔧 Solutions aux problèmes potentiels

### Problème : Versions incompatibles
**Solution 1** : Utiliser `requirements-locked.txt` (versions testées)
```bash
pip install -r requirements-locked.txt
```

**Solution 2** : Utiliser un environnement virtuel isolé
```bash
python -m venv venv_prof
venv_prof\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Problème : Fichiers manquants (modèle, index, embeddings)
**Solution** : Ces fichiers se génèrent automatiquement
```bash
python build_index.py
```
⏱️ Temps : 3-5 minutes

### Problème : Erreur FAISS sur Mac M1/M2
**Solution** : Utiliser conda au lieu de pip
```bash
conda install -c pytorch faiss-cpu
```

### Problème : PySpark nécessite Java
**Solution** : Installer Java JDK 8 ou 11
- Windows : https://adoptium.net/
- Linux : `sudo apt install openjdk-11-jdk`

Ou utiliser la version Pandas :
```bash
python prepare_corpus.py  # Au lieu de prepare_corpus_spark.py
```

## 📦 Ce que reçoit votre prof

### Sur GitHub :
1. **Code source complet** ✅
2. **Documentation détaillée** ✅
3. **Données d'entraînement** ✅ (sauf fichiers > 50MB)
4. **Scripts automatisés** ✅

### À générer localement (automatique) :
1. Modèle de sentence transformers (~90 MB)
2. Index FAISS (~quelques MB)
3. Embeddings (~quelques MB)

## 🎯 Instructions pour votre prof

Ajoutez ce texte dans votre email/soumission :

---

**Projet : Moteur de Recherche Sémantique Médical avec FAISS et Spark**

📌 **Lien GitHub** : https://github.com/OUEDRAOGOFred/Tp_spark_base_de_donn-e_vectorielle

### Installation rapide (3 commandes) :
```bash
git clone https://github.com/OUEDRAOGOFred/Tp_spark_base_de_donn-e_vectorielle.git
cd Tp_spark_base_de_donn-e_vectorielle
pip install -r requirements.txt
python build_index.py
streamlit run app_streamlit_v2.py
```

⏱️ **Temps total** : ~5-10 minutes (incluant téléchargement du modèle)

📖 **Documentation complète** : Voir `INSTALLATION.md` pour le guide détaillé

💡 **Note** : Certains fichiers volumineux (modèles, index) sont générés automatiquement lors de l'installation pour respecter les limites de GitHub.

---

## ✅ Recommandations finales

### Avant de soumettre :

1. **Tester sur une machine propre** (si possible)
   ```bash
   # Dans un nouveau dossier
   git clone https://github.com/OUEDRAOGOFred/Tp_spark_base_de_donn-e_vectorielle.git
   cd Tp_spark_base_de_donn-e_vectorielle
   pip install -r requirements.txt
   python build_index.py
   streamlit run app_streamlit_v2.py
   ```

2. **Vérifier que le README est clair**
   - ✅ Instructions d'installation
   - ✅ Captures d'écran (optionnel mais recommandé)
   - ✅ Description du projet
   - ✅ Technologies utilisées

3. **Ajouter un fichier d'informations projet** (optionnel)
   - Votre nom
   - Date de soumission
   - Version Python utilisée
   - Temps de développement

4. **Créer une archive ZIP de backup** (en plus de GitHub)
   ```bash
   # Inclure tout sauf .git et fichiers volumineux
   ```

## 🆘 Support

Si votre prof rencontre des problèmes :
- Consulter `INSTALLATION.md` - Section "Résolution de problèmes"
- Les fichiers volumineux exclus sont NORMAUX (voir `.gitignore`)
- Tous les fichiers manquants se génèrent automatiquement
- Compatibilité testée : Python 3.8-3.11, Windows/Linux/Mac

## 📊 Taille du projet

- **Sur GitHub** : ~20-30 MB (code + données essentielles)
- **Après installation complète** : ~150-200 MB (avec modèles)
- **Téléchargement automatique** : ~90 MB (modèle sentence-transformers)

---

✨ **Votre projet est prêt à être soumis !** ✨
