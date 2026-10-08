@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No se encontro el entorno Python. Consulta docs\OFFLINE.md.
  pause
  exit /b 1
)
if not exist "outputs\recorrido\project.json" (
  echo El analisis del recorrido todavia no esta preparado. Consulta docs\OFFLINE.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m src.offline serve --project "outputs\recorrido\project.json" --port 0
if errorlevel 1 pause
