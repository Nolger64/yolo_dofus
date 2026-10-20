#!/usr/bin/env python3
"""
Detector en tiempo real para Dofus en Arch Linux con GNOME Wayland.
Utiliza xdg-desktop-portal y PipeWire nativo para capturar la ventana del juego
sin pantallas negras, sin advertencias de fuentes Qt y con máximo rendimiento.
"""

import os
import sys

# Suprimir advertencias internas de Qt y OpenCV en GNOME Wayland
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false;qt.text.*=false;*.warning=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import time
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from pipewire_capture import PortalCapture, CaptureStream, is_available
    HAS_PIPEWIRE = is_available()
except ImportError:
    HAS_PIPEWIRE = False

# Colores para visualización (BGR)
# Personaje: Azul vivo | Enemigos: Rojo brillante
CLASS_COLORS = {
    "Personaje": (255, 140, 0),
    "personaje": (255, 140, 0),
    "Enemigos": (0, 0, 255),
    "enemigos": (0, 0, 255)
}
DEFAULT_COLOR = (0, 255, 0)

def draw_hud(frame, fps, count_personajes, count_enemigos, conf_thresh, paused):
    """Dibuja el panel superior con contadores y estado."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    status_str = "PAUSADO" if paused else f"{fps:4.1f} FPS"
    text = f"[{status_str}]  Personajes: {count_personajes} | Enemigos: {count_enemigos} | Conf: {conf_thresh:.2f}"
    cv2.putText(frame, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    shortcuts = "[P] Pausa | [S] Cambiar Ventana | [+/-] Conf | [Q] Salir"
    cv2.putText(frame, shortcuts, (max(15, w - 490), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

def find_model(model_arg):
    if os.path.exists(model_arg):
        return model_arg
    candidates = [
        "best.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    parser = argparse.ArgumentParser(description="Detector en vivo de Dofus (GNOME Wayland)")
    parser.add_argument("--model", type=str, default="best.pt", help="Ruta a los pesos (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza (0.1 a 0.95)")
    args = parser.parse_args()

    if not HAS_PIPEWIRE:
        print("[!] Error crítico: pipewire_capture no está disponible o el compositor no soporta screencast.")
        print("    Asegúrate de tener xdg-desktop-portal-gnome y pipewire instalados.")
        sys.exit(1)

    model_path = find_model(args.model)
    if not model_path:
        print(f"[!] No se encontró ningún modelo ({args.model} ni yolo11n.pt).")
        print("    Descarga tu 'best.pt' desde Google Colab y colócalo en esta carpeta:")
        print(f"    {os.path.abspath('.')}/best.pt")
        sys.exit(1)

    print(f"[+] Cargando modelo YOLO: {model_path}...")
    model = YOLO(model_path)

    portal = PortalCapture()
    session = None
    stream = None

    def start_window_capture():
        nonlocal session, stream
        if stream:
            try:
                stream.stop()
            except Exception:
                pass
        if session:
            try:
                session.close()
            except Exception:
                pass

        print("\n--------------------------------------------------------------")
        print("[*] Abriendo el selector nativo de GNOME Wayland...")
        print("    Haz clic en la ventana de Dofus y pulsa 'Compartir'.")
        print("--------------------------------------------------------------")
        
        session = portal.select_window()
        if not session:
            print("[!] Selección cancelada por el usuario.")
            return False

        print(f"[+] Ventana seleccionada: {session.width}x{session.height} px")
        # capture_interval=0.01 asegura captura fluida a máximos FPS
        stream = CaptureStream(session.fd, session.node_id, session.width, session.height, capture_interval=0.01)
        stream.start()
        return True

    if not start_window_capture():
        print("[!] No se seleccionó ninguna ventana. Saliendo...")
        return

    window_name = "Deteccion Dofus - GNOME Wayland"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(session.width, 1280), min(session.height, 720))

    conf_thresh = args.conf
    paused = False
    prev_time = time.time()
    fps = 0.0

    print("\n[+] Captura activa con PipeWire.")
    print("    Comandos en la ventana de detección:")
    print("      'q' o ESC : Salir")
    print("      's'       : Cambiar / re-seleccionar ventana")
    print("      'p'       : Pausar / Reanudar detección")
    print("      '+' o '=' : Subir confianza")
    print("      '-'       : Bajar confianza\n")

    try:
        while True:
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Verificar si la ventana seleccionada se cerró
            if stream.window_invalid:
                print("[!] La ventana seleccionada ya no es válida o fue cerrada.")
                if not start_window_capture():
                    break
                continue

            # Obtener el fotograma capturado
            frame_bgra = stream.get_frame()
            if frame_bgra is None:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            count_personajes = 0
            count_enemigos = 0

            if not paused:
                # Inferencia con YOLO
                results = model.predict(frame_bgr, conf=conf_thresh, imgsz=512, verbose=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if "personaje" in cls_name.lower():
                            count_personajes += 1
                        elif "enemigo" in cls_name.lower():
                            count_enemigos += 1

                        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

                        # Bounding box
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                        # Etiqueta
                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            draw_hud(frame_bgr, fps, count_personajes, count_enemigos, conf_thresh, paused)
            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('s'):
                start_window_capture()
            elif key == ord('p'):
                paused = not paused
            elif key in [ord('+'), ord('=')]:
                conf_thresh = min(0.95, conf_thresh + 0.05)
                print(f"[i] Confianza: {conf_thresh:.2f}")
            elif key == ord('-'):
                conf_thresh = max(0.10, conf_thresh - 0.05)
                print(f"[i] Confianza: {conf_thresh:.2f}")

    finally:
        if stream:
            stream.stop()
        if session:
            session.close()
        cv2.destroyAllWindows()
        print("[+] Detector cerrado correctamente.")

if __name__ == "__main__":
    main()
