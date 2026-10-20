#!/usr/bin/env python3
"""
Detector en tiempo real para Dofus usando YOLO y selección de ventana en Arch Linux / GNOME.
Permite seleccionar el área o ventana del juego, captura a altos FPS con MSS y dibuja las
detecciones de Personajes y Enemigos.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import mss
from ultralytics import YOLO

CONFIG_FILE = "window_config.json"

def load_saved_roi():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return data.get("roi")
        except Exception:
            pass
    return None

def save_roi(roi):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"roi": roi}, f, indent=2)
        print(f"[+] Coordenadas de ventana guardadas en {CONFIG_FILE}")
    except Exception as e:
        print(f"[!] No se pudo guardar configuración: {e}")

def select_window_roi():
    """
    Captura la pantalla completa y abre una interfaz para que el usuario
    arrastre un recuadro sobre la ventana de Dofus con el ratón.
    """
    print("\n[i] Capturando pantalla para selección de ventana...")
    with mss.mss() as sct:
        # Tomamos el monitor principal
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        raw_shot = sct.grab(mon)
        screen_img = np.array(raw_shot)
        screen_bgr = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

    h_orig, w_orig = screen_bgr.shape[:2]
    
    # Si la pantalla es 4K / HiDPI (como 3840x2400), escalamos para la vista de selección
    scale = 1.0
    if w_orig > 1920:
        scale = 1920.0 / w_orig
        preview = cv2.resize(screen_bgr, (int(w_orig * scale), int(h_orig * scale)))
    else:
        preview = screen_bgr.copy()

    window_title = "Selecciona la ventana de Dofus (Arrastra con el raton y pulsa ENTER o ESPACIO)"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    
    print("------------------------------------------------------------------")
    print(" 1. Arrastra con el ratón para encerrar la ventana del juego.")
    print(" 2. Presiona ENTER o BARRA ESPACIADORA para confirmar.")
    print(" 3. Presiona 'c' para cancelar.")
    print("------------------------------------------------------------------")
    
    r = cv2.selectROI(window_title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)

    x, y, w, h = r
    if w <= 10 or h <= 10:
        print("[!] Selección cancelada o área demasiado pequeña.")
        return None

    # Re-escalar coordenadas reales de la pantalla
    real_x = int(x / scale) + mon["left"]
    real_y = int(y / scale) + mon["top"]
    real_w = int(w / scale)
    real_h = int(h / scale)

    roi = {"left": real_x, "top": real_y, "width": real_w, "height": real_h}
    save_roi(roi)
    return roi

def draw_hud(frame, fps, count_personajes, count_enemigos, conf_thresh):
    """Dibuja un panel superior con estadísticas y controles."""
    h, w = frame.shape[:2]
    
    # Barra semitransparente superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Texto de estadísticas
    stats_text = f"FPS: {fps:4.1f} | Personajes: {count_personajes} | Enemigos: {count_enemigos} | Conf: {conf_thresh:.2f}"
    cv2.putText(frame, stats_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Ayuda rápida en esquina derecha
    help_text = "[S] Re-seleccionar | [P] Pausa | [+/-] Conf | [Q] Salir"
    cv2.putText(frame, help_text, (w - 480, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Detector en vivo de Dofus con YOLO")
    parser.add_argument("--model", type=str, default="best.pt", help="Ruta a los pesos del modelo (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza inicial (0.1 a 0.95)")
    parser.add_argument("--reselect", action="store_true", help="Forzar re-selección del área de la ventana")
    args = parser.parse_args()

    # Buscar el modelo si no existe el especificado
    model_path = args.model
    if not os.path.exists(model_path):
        # Buscar en rutas comunes
        alternatives = [
            "runs/detect/dofus_v3/weights/best.pt",
            "best.pt",
            "yolo11n.pt"
        ]
        found = False
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                found = True
                break
        if not found:
            print(f"[!] No se encontró '{args.model}'.")
            print("    Descarga el 'best.pt' desde Google Colab y colócalo en esta carpeta:")
            print(f"    {os.path.abspath('.')}/best.pt")
            print("    (Usando temporalmente 'yolo11n.pt' base para verificar la captura si está disponible)")
            if os.path.exists("yolo11n.pt"):
                model_path = "yolo11n.pt"
            else:
                sys.exit(1)

    print(f"[+] Cargando modelo: {model_path}...")
    model = YOLO(model_path)

    # Colores por clase (BGR)
    # Personaje -> Azul brillante (255, 120, 0)
    # Enemigos -> Rojo / Magenta (0, 0, 255)
    CLASS_COLORS = {
        "Personaje": (255, 120, 0),
        "personaje": (255, 120, 0),
        "Enemigos": (0, 0, 255),
        "enemigos": (0, 0, 255)
    }
    DEFAULT_COLOR = (0, 255, 0)

    # Cargar o pedir ROI
    roi = None
    if not args.reselect:
        roi = load_saved_roi()
        if roi:
            print(f"[+] Región de ventana cargada desde configuración: {roi}")

    if roi is None:
        roi = select_window_roi()
        if roi is None:
            print("[!] Operación abortada.")
            return

    cv2.namedWindow("Deteccion Dofus - YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deteccion Dofus - YOLO", min(roi["width"], 1280), min(roi["height"], 720))

    conf_thresh = args.conf
    paused = False
    prev_time = time.time()
    fps = 0.0

    print("\n[+] Captura iniciada. Enfócate en el juego.")
    print("    Comandos en la ventana de detección:")
    print("      'q' o ESC : Salir")
    print("      's'       : Re-seleccionar ventana de Dofus")
    print("      'p'       : Pausar / Continuar")
    print("      '+' o '=' : Aumentar confianza")
    print("      '-'       : Disminuir confianza\n")

    with mss.mss() as sct:
        while True:
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            if not paused:
                # 1. Capturar la región de la ventana
                raw_frame = sct.grab(roi)
                frame = np.array(raw_frame)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 2. Inferencia con YOLO
                results = model.predict(frame_bgr, conf=conf_thresh, imgsz=512, verbose=False)
                
                count_personajes = 0
                count_enemigos = 0

                # 3. Dibujar detecciones
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
                        
                        # Etiqueta con fondo
                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # 4. Dibujar HUD informativo
                draw_hud(frame_bgr, fps, count_personajes, count_enemigos, conf_thresh)
                cv2.imshow("Deteccion Dofus - YOLO", frame_bgr)
            else:
                # Si está pausado, solo esperar teclas
                pass

            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: # 'q' o ESC
                break
            elif key == ord('s'):
                new_roi = select_window_roi()
                if new_roi:
                    roi = new_roi
                    cv2.resizeWindow("Deteccion Dofus - YOLO", min(roi["width"], 1280), min(roi["height"], 720))
            elif key == ord('p'):
                paused = not paused
                print(f"[i] {'Pausado' if paused else 'Reanudado'}")
            elif key in [ord('+'), ord('=')]:
                conf_thresh = min(0.95, conf_thresh + 0.05)
                print(f"[i] Confianza: {conf_thresh:.2f}")
            elif key == ord('-'):
                conf_thresh = max(0.10, conf_thresh - 0.05)
                print(f"[i] Confianza: {conf_thresh:.2f}")

    cv2.destroyAllWindows()
    print("[+] Programa finalizado.")

if __name__ == "__main__":
    main()
