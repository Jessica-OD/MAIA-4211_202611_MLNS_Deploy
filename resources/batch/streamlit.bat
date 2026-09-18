@echo off
REM Ejecuta la app de Streamlit localmente.
REM Ajusta PROJECT_ROOT si este .bat se mueve de ubicación.

set PROJECT_ROOT=%~dp0..\..
set PYTHONPATH=%PROJECT_ROOT%

cd /d %PROJECT_ROOT%
streamlit run streamlit_app.py

pause
