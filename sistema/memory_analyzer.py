"""Analiza VRAM/RAM y ajusta batch automático."""
import psutil

def auto_batch(vram_gb=0, ram_gb=0):
    """Calcula batch para no saturar."""
    if vram_gb >= 12: return 64
    if vram_gb >= 8: return 32
    if vram_gb >= 6: return 16
    if ram_gb >= 16: return 16
    return 8

def get_vram():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except: pass
    return 0

if __name__ == "__main__":
    print(f"VRAM {get_vram():.1f}GB -> batch {auto_batch(get_vram(), psutil.virtual_memory().total/1024**3)}")
