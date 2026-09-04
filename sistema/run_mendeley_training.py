
import subprocess, sys, json, time
from pathlib import Path
import os

manifest = Path("dataset/manifests/mendeley_6rj76z6y3n_manifest.csv")
cache_root = Path("dataset/processed")
task = "successor_positions126"
out = Path("docs/analisis/fase2-entrenamiento")
device = "cuda"
batch_size = 8
epochs = 40
num_workers = 6

# Use parallel_controller if available
try:
    sys.path.insert(0, "sistema")
    # parallel_controller has relative imports, call directly
    import importlib.util, psutil
    # Simulate parallel_config without relative import issues
    vram = 0
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except: pass
    ram = psutil.virtual_memory().total/1024**3
    def auto_batch(vram_gb, ram_gb):
        if vram_gb >=12: return 64
        if vram_gb >=8: return 32
        if vram_gb >=6: return 16
        if ram_gb >=16: return 16
        return 8
    batch_auto = auto_batch(vram, ram)
    # Override to 8 per task spec
    batch_size = 8
    print(f"[parallel_controller] vram {vram:.1f}GB ram {ram:.1f}GB -> batch auto {batch_auto} -> usando {batch_size}")
    print(f"[parallel_controller] workers 6 P-cores + 8 E-cores =14, usando num_workers={num_workers} + prefetch 4 pin_memory cuda")
except Exception as e:
    print(f"[parallel_controller] fallback: {e}")

cmd = [
    sys.executable,
    "entrenador/core-pt/lsm/training/train_classifier.py",
    "--manifest", str(manifest),
    "--cache-root", str(cache_root),
    "--task", task,
    "--out", str(out),
    "--device", device,
    "--batch-size", str(batch_size),
    "--epochs", str(epochs),
    "--num-workers", str(num_workers),
]
print("[CMD]", " ".join(cmd))
env = os.environ.copy()
env["PYTHONPATH"] = "entrenador/core-pt" + os.pathsep + env.get("PYTHONPATH","")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd="C:/Users/riemann/Desktop/programa de señas")

progress_path = Path("C:/Users/riemann/Desktop/entrenamiento_progreso.txt")
log_path = Path("docs/analisis/fase2-entrenamiento/train.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

history = []
with open(log_path, "w", encoding="utf-8") as log:
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
        try:
            obj = json.loads(line.strip())
            if "epoch" in obj and "validation" in obj:
                history.append(obj)
                epoch = obj["epoch"]
                val_f1 = obj["validation"].get("macro_f1", 0)
                val_acc = obj["validation"].get("accuracy", 0)
                train_loss = obj["train"].get("loss", 0)
                if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                    msg = f"Epoca {epoch:02d}/{epochs} - train_loss {train_loss:.4f} - val_acc {val_acc:.4f} - val_macro_f1 {val_f1:.4f} - elapsed {obj.get('elapsed_seconds',0):.1f}s"
                    print(f"[PROGRESO] {msg}")
                    with open(progress_path, "a", encoding="utf-8") as pf:
                        pf.write(msg + "\n")
            if "warning" in obj:
                print(f"[WARN] {obj}")
        except: pass

proc.wait()
print(f"Training finished exit_code={proc.returncode}")

# Post-process: copy best.pt and metrics to fase3, generate accuracy_curve.png in fase4
import shutil, json

fase2 = Path("docs/analisis/fase2-entrenamiento")
fase3 = Path("docs/analisis/fase3-evaluacion")
fase4 = Path("docs/analisis/fase4-graficas")
fase3.mkdir(parents=True, exist_ok=True)
fase4.mkdir(parents=True, exist_ok=True)

if (fase2/"best.pt").exists():
    shutil.copy2(fase2/"best.pt", fase3/"best.pt")
    print(f"Copiado best.pt a {fase3}")
if (fase2/"metrics.json").exists():
    shutil.copy2(fase2/"metrics.json", fase3/"metrics.json")
    print(f"Copiado metrics.json a {fase3}")
    # Generate accuracy_curve.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = json.loads((fase2/"metrics.json").read_text(encoding="utf-8"))
        hist = data.get("history", [])
        if not hist:
            # try from log history
            hist = history
        epochs_vals = [h["epoch"] for h in hist]
        train_loss = [h["train"].get("loss",0) for h in hist]
        val_acc = [h["validation"].get("accuracy",0) for h in hist]
        val_f1 = [h["validation"].get("macro_f1",0) for h in hist]
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.set_xlabel("Epoca")
        ax1.set_ylabel("Loss / Accuracy")
        ax1.plot(epochs_vals, train_loss, label="train_loss", color="red", linestyle="--")
        ax1.plot(epochs_vals, val_acc, label="val_accuracy", color="blue")
        ax1.plot(epochs_vals, val_f1, label="val_macro_f1", color="green")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"mendeley_6rj76z6y3n successor_positions126 - {len(epochs_vals)} epocas - best val F1 {data.get('best_validation_macro_f1',0):.4f}")
        # Annotate best epoch
        best_epoch = max(hist, key=lambda x: x["validation"].get("macro_f1",0))["epoch"] if hist else 1
        ax1.axvline(best_epoch, color="gray", linestyle=":", label=f"best {best_epoch}")
        plt.tight_layout()
        out_png = fase4/"accuracy_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"Generada curva en {out_png}")
        # Also write progreso final
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(f"\nEntrenamiento completado. Best val macro F1: {data.get('best_validation_macro_f1',0):.4f} en epoca {best_epoch}\n")
            pf.write(f"Curva guardada en {out_png}\n")
            pf.write(f"best.pt guardado en {fase2}/best.pt y {fase3}/best.pt\n")
            pf.write(f"metrics guardado en {fase2}/metrics.json y {fase3}/metrics.json\n")
    except Exception as e:
        print(f"Error generando grafica: {e}")
        import traceback; traceback.print_exc()

print("Post-proceso completado")
