# ⚠️ IMPORTANT - Compatibilité Python 3.13

## 🔴 Problème détecté

Vous utilisez **Python 3.13.3** qui est très récent. Certaines bibliothèques du projet peuvent ne pas encore être entièrement compatibles.

## ✅ Solutions recommandées

### Option 1 : Installer Python 3.10 ou 3.11 (RECOMMANDÉ)

**Pour votre prof**, il est préférable d'utiliser Python 3.10 ou 3.11 pour éviter tout problème :

1. Télécharger Python 3.11 : https://www.python.org/downloads/release/python-3110/
2. Installer en parallèle (cocher "Add to PATH")
3. Créer un environnement virtuel :
   ```bash
   py -3.11 -m venv venv_prof
   venv_prof\Scripts\activate
   pip install -r requirements.txt
   ```

### Option 2 : Utiliser pyenv (pour gérer plusieurs versions)

```bash
# Installer pyenv-win
# Puis installer Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0
```

### Option 3 : Tester avec Python 3.13 (peut fonctionner)

Si vous voulez quand même tester avec 3.13 :

```bash
pip install -r requirements.txt
```

**Problèmes potentiels avec Python 3.13 :**
- `numpy` : Peut nécessiter une version >= 1.26
- `faiss-cpu` : Peut ne pas avoir de wheel précompilé
- `pyspark` : Compatibilité à vérifier

## 🎯 Recommandation finale

**Pour la soumission à votre prof :**

### Dans le README, ajoutez cette note :

```markdown
## 🐍 Versions Python Recommandées

**✅ Testé et fonctionnel :**
- Python 3.10.x
- Python 3.11.x

**⚠️ Non testé :**
- Python 3.12.x
- Python 3.13.x

Pour garantir la compatibilité, nous recommandons Python 3.10 ou 3.11.
```

## 📋 Checklist avant soumission

1. [ ] Tester le projet avec Python 3.10 ou 3.11
2. [ ] Mettre à jour PROJET_INFO.md avec la version Python testée
3. [ ] Ajouter une note dans README.md sur les versions compatibles
4. [ ] Optionnel : Créer un environnement conda pour isoler les dépendances

## 🔧 Commandes de test rapide

```bash
# Créer un environnement virtuel avec Python 3.11
python3.11 -m venv venv_test
venv_test\Scripts\activate  # Windows
source venv_test/bin/activate  # Linux/Mac

# Installer et tester
pip install -r requirements.txt
python build_index.py
streamlit run app_streamlit_v2.py
```

## 💡 Alternative : Docker

Pour une compatibilité maximale, vous pouvez créer un Dockerfile :

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app_streamlit_v2.py"]
```

Cela garantit que votre prof aura exactement le même environnement.
