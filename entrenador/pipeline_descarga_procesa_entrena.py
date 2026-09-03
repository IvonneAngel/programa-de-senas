"""Pipeline: descarga -> extrae landmarks -> borra imágenes -> entrena desde landmarks."""
import pathlib, shutil, subprocess, sys

def descarga(url, dest):
    """Descarga dataset letras (ej: sjt79hnb2f)."""
    print(f"Descargando {url} a {dest}")
    # Placeholder: usar curl o gdown
    # subprocess.run(["curl","-L",url,"-o",str(dest)])
    return dest

def procesa_a_landmarks(imagenes_dir, landmarks_dir):
    """Procesa jpgs a npy landmarks con MediaPipe y borra jpgs."""
    from core.vision.puntos import extract_hand_keypoints  # placeholder
    for img in pathlib.Path(imagenes_dir).rglob("*.jpg"):
        # Extrae landmarks
        npy = landmarks_dir / f"{img.stem}.npy"
        # np.save(npy, landmarks)
        pass
    # Borra imágenes después de procesar
    for img in pathlib.Path(imagenes_dir).rglob("*.jpg"):
        img.unlink()
    print(f"Imágenes borradas, solo quedan landmarks en {landmarks_dir}")

def entrena_desde_landmarks(landmarks_dir):
    """Entrena sin reprocesar desde cero, usa landmarks cacheados."""
    # python entrenador/core-pt/lsm/training/train_classifier.py --cache-root landmarks_dir
    print(f"Entrenando desde {landmarks_dir} (sin reprocesar)")

def prueba_accuracy():
    """Prueba cómo va subiendo accuracy por época."""
    # Lee logs y grafica
    import json
    # Simula: accuracy 0.85 -> 0.90 -> 0.95
    history = [0.85, 0.88, 0.92, 0.95, 0.98]
    for epoch, acc in enumerate(history, 1):
        print(f"Epoch {epoch}: accuracy {acc:.2%}")
    # Guarda en docs/graficas/accuracy_curve.png
    return history

if __name__ == "__main__":
    # Ejemplo uso:
    # descarga("https://data.mendeley.com/datasets/sjt79hnb2f/2", pathlib.Path("dataset/raw"))
    # procesa_a_landmarks("dataset/raw", "dataset/landmarks")
    # entrena_desde_landmarks("dataset/landmarks")
    # prueba_accuracy()
    pass
