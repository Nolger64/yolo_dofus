#!/bin/bash
# Script de inicio para el detector de Dofus en GNOME Wayland
cd "$(dirname "$0")"

# Silenciar mensajes y advertencias internas de Qt / OpenCV en GNOME Wayland
export QT_LOGGING_RULES="qt.qpa.*=false;qt.text.*=false;*.warning=false"
export OPENCV_LOG_LEVEL=ERROR

if [ ! -d ".venv" ]; then
    echo "Configurando entorno virtual..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv ultralytics pipewire-capture opencv-python
fi

echo "Iniciando detector de Dofus (GNOME Wayland)..."
.venv/bin/python detect_dofus.py "$@"
