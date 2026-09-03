"""Controla paralelismo DataLoader, MediaPipe, OpenCV. Config: 6 P-cores + 8 E-cores + RTX 4060, batch auto, prefetch 4."""
import os, multiprocessing

def get_workers():
    """Workers P-cores fuertes: 6 P-cores i7-13650HX."""
    import psutil
    physical = psutil.cpu_count(logical=False) or 14
    logical = psutil.cpu_count(logical=True) or 20
    if logical > physical:  # hybrid detectado
        return 6  # 6 P-cores fuertes (performance cores)
    return min(8, max(4, physical-2))

def get_mediapipe_workers():
    """E-cores para MediaPipe/OpenCV background: 8 E-cores."""
    import psutil
    physical = psutil.cpu_count(logical=False) or 14
    if physical >= 14:
        return 8  # 8 E-cores eficiencia
    return max(2, physical - 6)

def parallel_config():
    """Config para RTX 4060: batch auto, prefetch 4, pin_memory, persistent_workers."""
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
        "num_workers": get_workers(),  # 6 P-cores
        "mediapipe_workers": get_mediapipe_workers(),  # 8 E-cores
        "total_workers": get_workers() + get_mediapipe_workers(),  # 14
        "pin_memory": gpu["device"]=="cuda",
        "prefetch_factor": 4,
        "persistent_workers": True,
    }

if __name__ == "__main__":
    cfg = parallel_config()
    print(cfg)
    assert cfg["num_workers"] == 6, "P-cores deben ser 6"
    assert cfg["mediapipe_workers"] == 8, "E-cores deben ser 8"
    assert cfg["total_workers"] == 14, "total 6P+8E=14"
    print("[OK] 6P+8E=14 + RTX 4060 prefetch4")

