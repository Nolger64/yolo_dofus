# ⚔️ YOLO Dofus: Detección de Personajes y Enemigos en Tiempo Real

Este proyecto implementa un sistema de visión por computadora para la detección de **Personajes** y **Enemigos** en el videojuego **Dofus** utilizando arquitecturas **YOLO** (You Only Look Once), gestión de datasets mediante **Roboflow MCP**, y un detector en tiempo real adaptado para **Arch Linux** sobre **GNOME Wayland**.

---

## 👥 Estudiantes / Autores

* **Nolger Rodríguez**
* **Yoed Potillo**
* **Rafael Pereira**

---

## 📋 Resumen del Trabajo Realizado

### 1. Integración con Roboflow vía Model Context Protocol (MCP)
* Conexión directa del entorno de desarrollo con el servidor oficial alojado de Roboflow (`https://mcp.roboflow.com/mcp`).
* Autenticación mediante **API Key** en cabeceras HTTP seguras dentro de `~/.gemini/config/mcp_config.json`.
* Inspección programática de proyectos, metadatos, clases (`Personaje` y `Enemigos`) y versiones generadas.

### 2. Gestión y Exportación del Dataset (Versión 3)
* Generación y exportación de la **Versión 3** del dataset de Dofus:
  * **208 imágenes totales**: 149 para entrenamiento (*train*), 38 para validación (*valid*) y 21 para pruebas (*test*).
  * **Preprocesamiento:** Auto-orientación y redimensionamiento a `512x512` píxeles.
  * **Formato:** Exportación directa en estructura YOLOv8 / YOLO11 (`data.yaml` con anotaciones en coordenadas normalizadas).

### 3. Estrategia de Entrenamiento (CPU Local vs. Google Colab T4)
* **Intento inicial en nube de Roboflow:** Restringido por créditos de entrenamiento en la plataforma.
* **Entorno local:** Se configuró un entorno virtual con `uv` en Python 3.12 y Ultralytics. Aunque funcional en CPU (Intel Core i7-8569U), el tiempo de cómputo para 50 épocas superaba los 35-40 minutos.
* **Entrenamiento acelerado en Google Colab:** Se elaboró el cuaderno interactivo [`entrenamiento_yolo_dofus_colab.ipynb`](./entrenamiento_yolo_dofus_colab.ipynb) utilizando una **GPU NVIDIA T4** gratuita, reduciendo el tiempo de entrenamiento a **menos de 2 minutos** para 50 épocas con el modelo **YOLO11 Nano** (`yolo11n.pt`).

### 4. Detector en Vivo para Arch Linux + GNOME Wayland
* **Problema superado:** En GNOME con Wayland, los métodos de captura clásicos (MSS / X11) arrojan pantallas negras por restricciones de seguridad del compositor Mutter, además de conflictos con el backend Qt de OpenCV.
* **Solución nativa:** Implementación de [`detect_dofus.py`](./detect_dofus.py) mediante **PipeWire** y **XDG Desktop Portal** (`pipewire-capture`):
  * Selector nativo del sistema para elegir la ventana de Dofus.
  * Transmisión de fotogramas fluida a altos FPS sin pantallas negras.
  * Supresión de advertencias de fuentes Qt.

---

## 📊 Diagnóstico del Modelo y Métricas

A partir de la evaluación de la matriz de confusión obtenida tras el entrenamiento:

* **Aspectos positivos (Alto *Recall*):**
  * De los **Enemigos reales**, el modelo detectó correctamente **109** (solo 4 se le escaparon al fondo y 21 fueron confundidos con Personajes).
  * De los **Personajes reales**, detectó correctamente **26** (solo 4 se le escaparon al fondo y 8 fueron confundidos con Enemigos).
* **Puntos críticos (Falsos Positivos en Fondo):**
  * El modelo registró **1,023 falsos positivos de Enemigos** y **291 falsos positivos de Personajes** predichos sobre el `background` (fondo).
  * **Causa principal:** Etiquetado no exhaustivo (monstruos o personajes visibles en el juego que no fueron anotados en algunas capturas) sumado a un umbral de confianza de validación muy bajo por defecto (0.25).

---

## 🚀 Posibles Mejoras (Trabajo Futuro)

1. **Etiquetado Exhaustivo del Dataset:**
   * En juegos con vistas isométricas y muchos elementos como Dofus, es fundamental que si un monstruo o personaje aparece en una captura, esté debidamente anotado. Si no se marca, el modelo lo detectará pero el evaluador lo penalizará como falso positivo.
2. **Ajuste del Umbral de Confianza (*Confidence Threshold*):**
   * Configurar el filtro de inferencia en tiempo real en un rango de **0.55 a 0.65** (`--conf 0.60`). Esto elimina de inmediato la inmensa mayoría de las predicciones débiles sobre árboles, suelo e interfaz.
3. **Aumento de Datos (*Data Augmentation*):**
   * Incorporar variaciones de brillo, contraste y mapas oscuros/subterráneos en Roboflow para robustecer la detección frente a los distintos mapas de Dofus.
4. **Separación de Clases y Sprites:**
   * Distinguir entre clases específicas de monstruos o estados de combate para reducir las confusiones cruzadas entre personajes y enemigos.
5. **Más Épocas y Modelo Superior:**
   * Probar arquitecturas como **YOLO11s** (Small) o **YOLO11m** (Medium) a 100 épocas con *Early Stopping* para aumentar la precisión en mapas concurridos.

---

## 🎮 Guía de Uso del Detector en Vivo

### Requisitos del Sistema
* **SO:** Arch Linux (o cualquier distribución Linux moderna).
* **Entorno de escritorio:** GNOME (sesión Wayland recomendada) con `pipewire` y `xdg-desktop-portal-gnome`.
* **Python:** 3.11 o 3.12 (gestionado automáticamente con `uv`).

### Instrucciones de Ejecución

1. Coloca el archivo entrenado `best.pt` (descargado de Colab) en la raíz del proyecto.
2. Ejecuta el script de inicio:
   ```bash
   ./run_detector.sh
   ```
   *(Si deseas usar un umbral de confianza específico, por ejemplo 0.60)*:
   ```bash
   ./run_detector.sh --conf 0.60
   ```
3. Se abrirá la ventana nativa de GNOME: haz clic en la ventana de **Dofus** y pulsa **"Compartir"**.
4. ¡Listo! La ventana de visualización mostrará las cajas delimitadoras en tiempo real:
   * 🟦 **Personaje** (Azul)
   * 🟥 **Enemigos** (Rojo)

### Atajos de Teclado en Vivo
| Tecla | Acción |
|---|---|
| `q` o `ESC` | Salir del programa |
| `p` | Pausar / Reanudar detección |
| `s` | Cambiar o re-seleccionar la ventana |
| `+` / `=` | Aumentar umbral de confianza en tiempo real |
| `-` | Disminuir umbral de confianza en tiempo real |
