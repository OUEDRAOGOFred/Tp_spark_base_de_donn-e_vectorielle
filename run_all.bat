@echo off
echo ========================================================
echo  Medical Semantic Search - Pipeline Complet
echo ========================================================
echo.
echo Ce script va executer toutes les etapes du projet:
echo   1. Preparation du corpus (Pandas OU Spark)
echo   2. Construction de l'index FAISS
echo   3. Evaluation du systeme
echo.

echo CHOIX DE LA VERSION:
echo   [1] Pandas - Rapide et simple (recommande pour demo)
echo   [2] Spark - Big Data, scalable (titre du projet!)
echo.
set /p choice="Votre choix (1 ou 2): "

if "%choice%"=="2" (
    set CORPUS_SCRIPT=prepare_corpus_spark.py
    echo.
    echo =^> Utilisation de Apache Spark (Big Data)
) else (
    set CORPUS_SCRIPT=prepare_corpus.py
    echo.
    echo =^> Utilisation de Pandas (rapide)
)

pause

echo.
echo ========================================================
echo  ETAPE 1/3 - Preparation du Corpus
echo ========================================================
echo.
python %CORPUS_SCRIPT%
if errorlevel 1 (
    echo ERREUR lors de la preparation du corpus!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  ETAPE 2/3 - Construction de l'Index FAISS
echo ========================================================
echo.
python build_index.py
if errorlevel 1 (
    echo ERREUR lors de la construction de l'index!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  ETAPE 3/3 - Evaluation du Systeme
echo ========================================================
echo.
python evaluate_search.py
if errorlevel 1 (
    echo ERREUR lors de l'evaluation!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  PIPELINE TERMINE AVEC SUCCES!
echo ========================================================
echo.
echo Fichiers generes:
echo   - docs_medical.csv
echo   - embeddings_medical.npy
echo   - medical_faiss.index
echo   - index_metadata.pkl
echo   - evaluation_*.csv
echo   - evaluation_*.png
echo.
echo Vous pouvez maintenant lancer:
echo   - L'API: start_api.bat
echo   - L'interface: start_app.bat
echo.
pause
