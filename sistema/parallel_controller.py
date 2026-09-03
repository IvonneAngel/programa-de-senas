"""Controla paralelismo DataLoader, MediaPipe, OpenCV."""
import os, multiprocessing

def get_workers():
    """Workers óptimos: usa P-cores fuertes (no solo 2)."""
    import psutil
    physical = psutil.cpu_count(logical=False) or 6
    logical = psutil.cpu_count(logical=True) or 12
    # Detecta hybrid: si logical > physical, hay E-cores
    if logical > physical:
        p_cores = 6  # i7-13650HX: 6 P-cores fuertes
        return p_cores
    return min(8, max(4, physical-2))

def get_mediapipe_workers():
    """E-cores para MediaPipe/OpenCV background."""
    import psutil
    physical = psutil.cpu_count(logical=False) or 6
    return max(2, physical - 6)  # 8 E-cores -> 4 workers

def parallel_config():
    """Config para entrenamiento rápido."""
    from .gpu_detector import detect_gpu
    from .memory_analyzer import auto_batch, get_vram
    import psutil
    gpu = detect_gpu()
    vram = get_vram()
    ram = psutil.virtual_memory().total/1024**3
    return {
        "device": gpu["device"],
        "priority": gpu["priority"],
        "batch_size": auto_batch(vram, ram),
        "num_workers": get_workers(),  # 6 P-cores fuertes
        "mediapipe_workers": get_mediapipe_workers(),  # 4 E-cores
        "pin_memory": gpu["device"]=="cuda",
        "prefetch_factor": 4,  # subido de 2 a 4 para RTX
        "persistent_workers": True,
    }

if __name__ == "__main__":
    print(parallel_config())
