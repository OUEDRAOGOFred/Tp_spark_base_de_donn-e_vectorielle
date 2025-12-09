@echo off
echo ================================================
echo  Medical Semantic Search - Interface Streamlit
echo ================================================
echo.

echo [1/2] Activation de l'environnement...
REM Si vous utilisez un environnement virtuel, décommentez la ligne suivante:
REM call venv\Scripts\activate

echo [2/2] Demarrage de Streamlit...
echo.
echo L'application sera accessible sur: http://localhost:8501
echo.

streamlit run app_streamlit_v2.py

pause
