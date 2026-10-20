#!/bin/bash
# Script de inicio rápido para el detector de Dofus
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Entorno virtual no encontrado. Creando..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv ultralytics mss supervision opencv-python
fi

echo "Iniciando detector de Dofus..."
.venv/bin/python detect_dofus.py "$@"
