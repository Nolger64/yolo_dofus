import os
import torch
from ultralytics import YOLO

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--> Dispositivo seleccionado: {device}")
    
    # Cargamos el modelo base YOLO11 Nano pre-entrenado
    model = YOLO("yolo11n.pt")
    
    data_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_v3", "data.yaml")
    print(f"--> Usando dataset: {data_yaml}")
    
    # Iniciar entrenamiento
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=512,
        batch=16,
        workers=4,
        device=device,
        project="runs/detect",
        name="dofus_v3",
        exist_ok=True,
        verbose=True
    )
    
    print("\n[OK] Entrenamiento finalizado con éxito.")
    print(f"Pesos guardados en: runs/detect/dofus_v3/weights/best.pt")

if __name__ == "__main__":
    main()
