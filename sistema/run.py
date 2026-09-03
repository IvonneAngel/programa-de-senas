"""Sistema RTX priority + batch auto + paralelismo."""
from .gpu_detector import detect_gpu
from .memory_analyzer import auto_batch, get_vram
from .parallel_controller import parallel_config
import psutil

def main():
    gpu = detect_gpu()
    print(f"GPU: {gpu}")
    cfg = parallel_config()
    print(f"Config: {cfg}")
    # Aquí se lanzaría entrenamiento con cfg
    return cfg

if __name__ == "__main__":
    main()
