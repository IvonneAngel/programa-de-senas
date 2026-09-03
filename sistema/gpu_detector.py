"""Detecta GPU RTX/CUDA y prioriza."""
import subprocess

def detect_gpu():
    """Detecta GPU, prioriza RTX."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "RTX" in name:
                return {"device": "cuda", "name": name, "priority": "RTX"}
            return {"device": "cuda", "name": name, "priority": "CUDA"}
    except: pass
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if out.returncode==0 and "RTX" in out.stdout:
            return {"device": "cuda", "name": out.stdout.strip(), "priority": "RTX"}
    except: pass
    return {"device": "cpu", "name": "CPU", "priority": "CPU"}

if __name__ == "__main__":
    print(detect_gpu())
