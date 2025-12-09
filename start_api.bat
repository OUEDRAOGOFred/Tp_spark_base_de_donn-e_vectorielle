@echo off
echo ================================================
echo  Medical Semantic Search - API FastAPI
echo ================================================
echo.

echo [1/2] Activation de l'environnement...
REM Si vous utilisez un environnement virtuel, décommentez la ligne suivante:
REM call venv\Scripts\activate

echo [2/2] Demarrage de l'API...
echo.
echo L'API sera accessible sur: http://localhost:8000
echo Documentation Swagger: http://localhost:8000/docs
echo.

python api_medical_v2.py

pause
