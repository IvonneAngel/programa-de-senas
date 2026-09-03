"""Sistema RTX priority + batch auto + paralelismo 6P+8E prefetch4."""
from .gpu_detector import detect_gpu
from .memory_analyzer import auto_batch, get_vram
from .parallel_controller import parallel_config
import psutil

def main():
    gpu = detect_gpu()
    print(f"GPU: {gpu}")
    cfg = parallel_config()
    print(f"Config: {cfg} total={cfg['total_workers']} (6P+8E) prefetch={cfg['prefetch_factor']}")
    # lanza auto_extract automatico
    from .auto_extract import main as auto_main
    auto_main()
    return cfg

if __name__ == "__main__":
    main()
